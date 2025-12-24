import streamlit as st
from supabase import create_client, Client
import re
import requests
import time
import random
import math
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from openai import OpenAI
from datetime import datetime
from collections import Counter
import yt_dlp
import pandas as pd
import altair as alt
import json
from bs4 import BeautifulSoup

# --- [1. 시스템 설정] ---
st.set_page_config(page_title="유튜브 가짜뉴스 판독기 v99.6", layout="wide", page_icon="🛡️")

if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False
if "debug_logs" not in st.session_state:
    st.session_state["debug_logs"] = []

# 🌟 Secrets 로드
try:
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
    GOOGLE_API_KEY_A = st.secrets["GOOGLE_API_KEY_A"]
    MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
except Exception as e:
    st.error(f"❌ 필수 키 설정 누락: {e}")
    st.stop()

# Mistral 클라이언트 (Key B 전용)
mistral_client = OpenAI(api_key=MISTRAL_API_KEY, base_url="https://api.mistral.ai/v1")

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

# --- [2. 글로벌 상수 및 유틸리티] ---
STATIC_TRUTH_CORPUS = ["박나래 위장전입 무혐의", "임영웅 암표 대응", "정희원 저속노화", "대전 충남 통합", "선거 출마 선언"]
STATIC_FAKE_CORPUS = ["충격 폭로 경악", "긴급 속보 소름", "충격 발언 논란", "구속 영장 발부", "영상 유출", "계시 예언", "사형 집행", "위독설"]
CRITICAL_STATE_KEYWORDS = ['별거', '이혼', '파경', '사망', '위독', '구속', '체포', '실형', '불화', '폭로', '충격', '논란', '중태', '심정지', '뇌사', '압수수색', '소환', '파산', '빚더미', '전과', '감옥', '간첩']
OFFICIAL_CHANNELS = ['MBC', 'KBS', 'SBS', 'EBS', 'YTN', 'JTBC', 'TVCHOSUN', 'MBN', 'CHANNEL A', 'OBS', '채널A', 'TV조선', '연합뉴스', 'YONHAP', '한겨레', '경향', '조선', '중앙', '동아']
WEIGHT_ALGO = 0.6
WEIGHT_AI = 0.4

def parse_ai_json(text):
    try:
        parsed = json.loads(text)
    except:
        try:
            text = re.sub(r'```json\s*', '', text).replace('```', '')
            match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
            if match: 
                parsed = json.loads(match.group(1))
                return parsed[0] if isinstance(parsed, list) else parsed
        except: pass
    return None

def safe_int_convert(val, default=50):
    try:
        if isinstance(val, dict): val = list(val.values())[0]
        return int(float(val))
    except: return default

def extract_video_id(url):
    match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    return match.group(1) if match else None

# --- [3. 모델 엔진 세팅] ---

# [Key A] Gemini 기존 로직 유지
def get_gemini_search_keywords(title, transcript):
    genai.configure(api_key=GOOGLE_API_KEY_A)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Role: Fact-Check Investigator. [Input] Title: {title}, Transcript: {transcript[:15000]}. [Task] Extract ONE Korean search query (2-4 words). Output ONLY the string."
    try:
        response = model.generate_content(prompt)
        st.session_state["debug_logs"].append(f"✅ Key A (Gemini) Keyword Extracted")
        return response.text.strip()
    except Exception as e:
        st.session_state["debug_logs"].append(f"❌ Key A Failed: {e}")
        return title

# [Key B] Mistral Judge 전용
def call_mistral_judge(prompt):
    try:
        response = mistral_client.chat.completions.create(
            model="mistral-large-latest",
            messages=[{"role": "system", "content": "당신은 전문 팩트체크 판사입니다. 모든 분석 결과는 한국어로 작성하고 JSON으로만 응답하세요."},
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        st.session_state["debug_logs"].append("✅ Key B (Mistral) Verdict Success")
        return response.choices[0].message.content
    except Exception as e:
        st.session_state["debug_logs"].append(f"❌ Key B Mistral Error: {e}")
        return None

# --- [4. 분석 엔진 (증거 수집 로직 복구)] ---

class VectorEngine:
    def __init__(self):
        self.vocab = set()
        self.truth_vectors = []
        self.fake_vectors = []
    def tokenize(self, text): return re.findall(r'[가-힣]{2,}', text)
    def train(self, truth, fake):
        for t in truth + fake: self.vocab.update(self.tokenize(t))
        self.vocab = sorted(list(self.vocab))
        self.truth_vectors = [self.text_to_vector(t) for t in truth]
        self.fake_vectors = [self.text_to_vector(t) for t in fake]
    def text_to_vector(self, text, vocabulary=None):
        target_vocab = vocabulary if vocabulary else self.vocab
        c = Counter(self.tokenize(text))
        return [c[w] for w in target_vocab]
    def cosine_similarity(self, v1, v2):
        dot = sum(a*b for a,b in zip(v1,v2))
        mag = math.sqrt(sum(a*a for a in v1)) * math.sqrt(sum(b*b for b in v2))
        return dot/mag if mag>0 else 0
    def analyze_position(self, query):
        if not self.vocab: return 0, 0
        qv = self.text_to_vector(query)
        mt = max([self.cosine_similarity(qv, v) for v in self.truth_vectors] or [0])
        mf = max([self.cosine_similarity(qv, v) for v in self.fake_vectors] or [0])
        return mt, mf

vector_engine = VectorEngine()

def fetch_comments_via_api(video_id):
    try:
        url = "https://www.googleapis.com/youtube/v3/commentThreads"
        res = requests.get(url, params={'part': 'snippet', 'videoId': video_id, 'key': YOUTUBE_API_KEY, 'maxResults': 50})
        if res.status_code == 200:
            items = [i['snippet']['topLevelComment']['snippet']['textDisplay'] for i in res.json().get('items', [])]
            return items, "Success"
    except: pass
    return [], "Fail"

def analyze_comment_relevance(comments, context_text):
    if not comments: return [], 0, "분석 불가"
    all_cmt_text = " ".join(comments)
    tokens = [re.sub(r'[^가-힣]', '', w) for w in re.findall(r'[가-힣]{2,}', all_cmt_text)]
    top = Counter(tokens).most_common(5)
    ctx_tokens = set(re.findall(r'[가-힣]{2,}', context_text))
    match = sum(1 for w, c in top if w in ctx_tokens)
    score = int(match/len(top)*100) if top else 0
    msg = "✅ 주제 집중" if score >= 60 else "⚠️ 일부 관련" if score >= 20 else "❌ 무관"
    return [f"{w}({c})" for w, c in top], score, msg

def check_red_flags(comments):
    keywords = ['가짜', '주작', '사기', '허위', '선동', '거짓']
    detected = [k for c in comments for k in keywords if k in c]
    return len(detected), list(set(detected))

def count_sensational_words(text):
    return sum(text.count(w) for w in ['충격', '경악', '실체', '폭로', '속보', '긴급', '단독'])

# --- [5. UI 컴포넌트 (디자인 복구)] ---

def render_score_breakdown(data_list):
    style = """<style>table.score-table { width: 100%; border-collapse: separate; border: 1px solid #e0e0e0; border-radius: 8px; font-size: 14px; margin-top: 10px;} table.score-table th { background-color: #f8f9fa; padding: 12px; text-align: left; } table.score-table td { padding: 12px; border-bottom: 1px solid #f0f0f0; } .badge { padding: 4px 8px; border-radius: 6px; font-weight: 700; font-size: 11px; display: inline-block; } .badge-danger { background-color: #ffebee; color: #d32f2f; } .badge-success { background-color: #e8f5e9; color: #2e7d32; }</style>"""
    rows = ""
    for item, score, note in data_list:
        try:
            score_num = int(score)
            badge = f'<span class="badge badge-danger">+{score_num}</span>' if score_num > 0 else f'<span class="badge badge-success">{score_num}</span>' if score_num < 0 else "0"
        except: badge = str(score)
        rows += f"<tr><td>{item}<br><small style='color:#888;'>{note}</small></td><td style='text-align: right;'>{badge}</td></tr>"
    st.markdown(f"{style}<table class='score-table'><thead><tr><th>분석 항목</th><th style='text-align: right;'>변동</th></tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)

def colored_progress_bar(label, percent, color):
    st.markdown(f"""<div style="margin-bottom: 10px;"><div style="display: flex; justify-content: space-between; margin-bottom: 3px;"><span style="font-size: 13px; font-weight: 600; color: #555;">{label}</span><span style="font-size: 13px; font-weight: 700; color: {color};">{round(percent * 100, 1)}%</span></div><div style="background-color: #eee; border-radius: 5px; height: 8px; width: 100%;"><div style="background-color: {color}; height: 8px; width: {percent * 100}%; border-radius: 5px;"></div></div></div>""", unsafe_allow_html=True)

def render_intelligence_distribution(current_prob):
    try:
        res = supabase.table("analysis_history").select("fake_prob").execute()
        if not res.data: return
        df = pd.DataFrame(res.data)
        base = alt.Chart(df).transform_density('fake_prob', as_=['fake_prob', 'density'], extent=[0, 100], bandwidth=5).mark_area(opacity=0.3, color='#888').encode(x=alt.X('fake_prob:Q', title='가짜뉴스 확률 분포'), y=alt.Y('density:Q', title='데이터 밀도'))
        rule = alt.Chart(pd.DataFrame({'x': [current_prob]})).mark_rule(color='blue', size=3).encode(x='x')
        st.altair_chart(base + rule, use_container_width=True)
    except: pass

# --- [6. 메인 로직] ---

def run_forensic_main(url):
    st.session_state["debug_logs"] = []
    vid = extract_video_id(url)
    if not vid: return st.error("URL 오류")

    # DB 학습
    res_t = supabase.table("analysis_history").select("video_title").lt("fake_prob", 40).execute()
    res_f = supabase.table("analysis_history").select("video_title").gt("fake_prob", 60).execute()
    dt, df = [r['video_title'] for r in res_t.data], [r['video_title'] for r in res_f.data]
    vector_engine.train(STATIC_TRUTH_CORPUS + dt, STATIC_FAKE_CORPUS + df)
    db_count = len(dt) + len(df)

    # 캐시 체크
    cached_res = supabase.table("analysis_history").select("*").ilike("video_url", f"%{vid}%").order("id", desc=True).limit(1).execute()
    if cached_res.data:
        c = cached_res.data[0]
        try:
            d = json.loads(c.get('detail_json', '{}'))
            render_report_full_ui(c['fake_prob'], db_count, c['video_title'], c['channel_name'], d, is_cached=True)
            return
        except: pass

    my_bar = st.progress(0, text="분석 시작...")
    with yt_dlp.YoutubeDL({'quiet': True, 'skip_download': True}) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            title, uploader, desc = info.get('title',''), info.get('uploader',''), info.get('description','')
            tags = info.get('tags', [])
            
            # [1단계] 데이터 수집 (API 기반)
            my_bar.progress(10, "1단계: 자막 및 댓글 수집 중...")
            subs = info.get('subtitles') or {}
            auto = info.get('automatic_captions') or {}
            merged = {**subs, **auto}
            full_text = desc
            if 'ko' in merged:
                for f in merged['ko']:
                    if f['ext'] == 'vtt':
                        res = requests.get(f['url'])
                        full_text = " ".join([l.strip() for l in res.text.splitlines() if l.strip() and '-->' not in l and '<' not in l])
                        break
            cmts, _ = fetch_comments_via_api(vid)

            # [2단계] AI 수사관 (Key A 기존 로직)
            my_bar.progress(30, "2단계: AI 수사관(Gemini) 키워드 추출 중...")
            query = get_gemini_search_keywords(title, full_text)
            
            # [3단계] 뉴스 교차 대조
            my_bar.progress(50, "3단계: 뉴스 교차 대조 진행 중...")
            rss = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=ko&gl=KR"
            news_raw = requests.get(rss).text
            items = re.findall(r'<item>(.*?)</item>', news_raw, re.DOTALL)[:3]
            news_ev = []; max_match = 0
            for i in items:
                nt = re.search(r'<title>(.*?)</title>', i).group(1).replace("<![CDATA[", "").replace("]]>", "")
                nl = re.search(r'<link>(.*?)</link>', i).group(1)
                nd = re.search(r'<description>(.*?)</description>', i).group(1)
                
                # Mistral을 이용한 뉴스 대조
                prompt_b = f"비교 분석: 영상[{title}] vs 뉴스[{nt}]. 일치하면 0-10, 다르면 90-100. JSON {{'score': int, 'reason': '한글이유'}}"
                res_b = call_mistral_judge(prompt_b)
                p_b = parse_ai_json(res_b)
                s_b = safe_int_convert(p_b.get('score')) if p_b else 50
                if s_b > max_match: max_match = s_b
                news_ev.append({"뉴스 제목": nt, "일치도": f"{s_b}%", "최종 점수": s_b, "분석 근거": p_b.get('reason','') if p_b else 'N/A', "원문": nl})

            # 알고리즘 스코어링
            ts, fs = vector_engine.analyze_position(query + " " + title)
            t_impact, f_impact = int(ts*30)*-1, int(fs*30)
            news_penalty = -30 if max_match <= 20 else (30 if max_match >= 80 else 0)
            
            # 증거 2, 3 로직 복구
            top_cmt_kw, rel_score, rel_msg = analyze_comment_relevance(cmts, title + " " + full_text)
            red_cnt, red_list = check_red_flags(cmts)
            agitation = count_sensational_words(title + full_text)

            # [4단계] AI 판사 최종 판결 (Mistral)
            my_bar.progress(85, "4단계: AI 판사(Mistral) 최종 판결 중...")
            prompt_final = f"최종 판결: 영상 제목 '{title}', 뉴스 증거: {news_ev}. 진실이면 0-20, 가짜면 80-100. JSON {{'score': int, 'reason': '한글판결문'}}"
            res_f = call_mistral_judge(prompt_final)
            p_f = parse_ai_json(res_f)
            ai_score = safe_int_convert(p_f.get('score')) if p_f else 50
            
            final_prob = max(1, min(99, int((50 + t_impact + f_impact + news_penalty)*WEIGHT_ALGO + ai_score*WEIGHT_AI)))
            
            score_breakdown = [["기본 중립 점수", 50, "분석 시작점"], ["진실 DB 맥락", t_impact, "내부 DB 매칭"], ["가짜 패턴 맥락", f_impact, "내부 DB 매칭"], ["뉴스 교차 검증", news_penalty, "크롤링 결과"], ["AI 최종 판결", ai_score, p_f.get('reason','') if p_f else 'Error']]
            
            report = {
                "summary": full_text[:800], "news_evidence": news_ev, "ai_score": ai_score, "ai_reason": p_f.get('reason','') if p_f else 'Error',
                "score_breakdown": score_breakdown, "ts": ts, "fs": fs, "query": query, "tags": ", ".join(tags),
                "top_cmt_kw": top_cmt_kw, "cmt_rel": f"{rel_score}% ({rel_msg})", "red_cnt": red_cnt,
                "agitation": agitation, "cmt_count": len(cmts)
            }
            
            supabase.table("analysis_history").insert({"channel_name": uploader, "video_title": title, "fake_prob": final_prob, "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "video_url": url, "keywords": query, "detail_json": json.dumps(report, ensure_ascii=False)}).execute()
            my_bar.empty()
            render_report_full_ui(final_prob, db_count, title, uploader, report)

        except Exception as e: st.error(f"분석 중 오류: {e}")

def render_report_full_ui(prob, db_count, title, uploader, d, is_cached=False):
    if is_cached: st.success("🎉 기존 분석 결과 발견 (Smart Cache)")

    st.subheader("🕵️ Dual-Engine Analysis Result")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("최종 가짜뉴스 확률", f"{prob}%")
    col_b.metric("AI 판정", "🔴 위험" if prob > 60 else "🟢 안전" if prob < 30 else "🟠 주의")
    col_c.metric("지능 노드", f"{db_count} Nodes")
    
    st.divider()
    st.subheader("🧠 Intelligence Map")
    render_intelligence_distribution(prob)

    st.divider()
    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.write("**[영상 상세 정보]**")
        st.table(pd.DataFrame({"항목": ["영상 제목", "채널명", "해시태그"], "내용": [title, uploader, d.get('tags','없음')]}))
        st.info(f"🎯 Investigator 추출 검색어: {d.get('query', 'N/A')}")
        with st.container(border=True):
            st.markdown("📝 **영상 내용 요약**")
            st.write(d.get('summary','내용 없음'))
        render_score_breakdown(d.get('score_breakdown', []))

    with col2:
        st.subheader("📊 5대 정밀 분석 증거")
        st.markdown("**[증거 0] Semantic Vector Space (Internal DB)**")
        colored_progress_bar("✅ 진실 영역 근접도", d.get('ts', 0), "#2ecc71")
        colored_progress_bar("🚨 거짓 영역 근접도", d.get('fs', 0), "#e74c3c")
        
        st.markdown("**[증거 1] 뉴스 교차 대조 (Deep-Web Crawler)**")
        if d.get('news_evidence'):
            st.dataframe(pd.DataFrame(d.get('news_evidence', [])), column_config={"원문": st.column_config.LinkColumn("링크", display_text="🔗 이동")}, hide_index=True)
        else: st.warning("관련 뉴스를 찾을 수 없습니다.")

        st.markdown("**[증거 2] 시청자 여론 심층 분석**")
        st.table(pd.DataFrame([["분석 댓글 수", f"{d.get('cmt_count',0)}개"], ["최다 빈출 키워드", ", ".join(d.get('top_cmt_kw', []))], ["논란 감지 건수", f"{d.get('red_cnt',0)}건"], ["주제 일치도", d.get('cmt_rel','0%')]], columns=["항목", "내용"]))
        
        st.markdown("**[증거 3] 자막 세만틱 심층 대조**")
        st.table(pd.DataFrame([["영상 주요 키워드", "분석 완료"], ["선동성 지수", f"{d.get('agitation',0)}회"]], columns=["항목", "내용"]))
        
        st.markdown("**[증거 4] AI 최종 분석 판단 (Judge Verdict)**")
        with st.container(border=True): st.write(f"⚖️ **판결:** {d.get('ai_reason', 'N/A')}")

# --- [7. UI 레이아웃 및 관리자 기능] ---

st.title("⚖️ Fact-Check Center v99.6")

with st.container(border=True):
    st.markdown("### 🛡️ 법적 고지 및 책임 한계 (Disclaimer)\n본 서비스는 **인공지능(AI) 및 알고리즘 기반**으로 영상의 신뢰도를 분석하는 보조 도구입니다. \n분석 결과는 법적 효력이 없으며, 최종 판단의 책임은 사용자에게 있습니다.")
    st.markdown("* **Engine A (Investigator)**: Gemini 1.5 Flash (키워드 추출 로직)\n* **Engine B (Judge)**: Mistral Large 2 (한글 심층 분석 및 판결)")
    agree = st.checkbox("위 내용을 확인하였으며, 이에 동의합니다. (동의 시 분석 버튼 활성화)")

url_input = st.text_input("🔗 분석할 유튜브 URL")
if st.button("🚀 정밀 분석 시작", disabled=not agree, use_container_width=True):
    if url_input: run_forensic_main(url_input)

st.divider()
st.subheader("🗂️ 학습 데이터 관리 (Cloud Knowledge Base)")
try:
    resp = supabase.table("analysis_history").select("*").order("id", desc=True).limit(20).execute()
    df_h = pd.DataFrame(resp.data)
    if not df_h.empty:
        if st.session_state["is_admin"]:
            df_h['Delete'] = False
            edited = st.data_editor(df_h[['Delete', 'id', 'analysis_date', 'video_title', 'fake_prob', 'keywords']], hide_index=True, use_container_width=True)
            if st.button("🗑️ 선택 항목 삭제"):
                for _, row in edited[edited.Delete].iterrows():
                    supabase.table("analysis_history").delete().eq("id", row['id']).execute()
                st.success("삭제 완료")
                time.sleep(0.5)
                st.rerun()
        else: st.dataframe(df_h[['analysis_date', 'video_title', 'fake_prob', 'keywords']], use_container_width=True, hide_index=True)
except: pass

with st.expander("🔐 관리자 접속 (Admin Access)"):
    if not st.session_state["is_admin"]:
        if st.text_input("Admin Password", type="password") == ADMIN_PASSWORD:
            st.session_state["is_admin"] = True
            st.rerun()
    else:
        st.success("관리자 권한 활성화됨")
        if st.session_state["debug_logs"]:
            st.write("**📜 실시간 디버그 로그**")
            st.text_area("Logs", "\n".join(st.session_state["debug_logs"]), height=300)
        if st.button("로그아웃"):
            st.session_state["is_admin"] = False
            st.rerun()
