import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from datetime import datetime, date
import seaborn as sns
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="기간별 워드클라우드 분석기",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 한글 폰트 설정
plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']

class StreamlitWordCloudAnalyzer:
    def __init__(self):
        # 확장된 불용어 리스트
        self.korean_stopwords = {
            '이', '그', '저', '것', '들', '는', '은', '을', '를', '에', '의', '가', '과', '와', '도', '만',
            '이다', '있다', '되다', '하다', '같다', '보다', '더', '매우', '정말', '아주', '조금', '좀',
            '때문', '위해', '통해', '대해', '관해', '따라', '위한', '통한', '대한', '관한', '따른',
            '등', '및', '또는', '그리고', '하지만', '그러나', '따라서', '그러므로', '즉', '예를',
            '수', '개', '명', '건', '점', '번', '차', '회', '년', '월', '일', '시', '분', '초',
            '모든', '각', '어떤', '여러', '다른', '새로운', '주요', '중요', '특별', '일반',
            '방법', '시스템', '장치', '기술', '연구', '분석', '결과', '효과', '성능', '특성','포함하는','치환된'
        }
        
        # 확장된 영어 불용어 리스트 (생물학/의학 용어 추가)
        self.english_stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall',
            'method', 'system', 'apparatus', 'device', 'present', 'invention', 'embodiment', 'according',
            'comprising', 'including', 'wherein', 'therefor', 'thereof', 'herein', 'said', 'such',
            'provided', 'configured', 'adapted', 'disposed', 'formed', 'made', 'using', 'based',
            'this', 'that', 'these', 'those', 'which', 'who', 'whom', 'whose',
            'it', 'its', 'they', 'their', 'them', 'he', 'his', 'she', 'her', 'we', 'our', 'us',
            'i', 'me', 'my', 'you', 'your', 'yours', 'he', 'him', 'she', 'her', 'they', 'them',
            'there', 'here', 'where', 'when', 'why', 'how', 'one', 'two', 'three', 'four', 'five',
            'first', 'second', 'third', 'next', 'last', 'many','sub','cell','cells','human','humans',
            'patient', 'patients', 'study', 'studies', 'analysis', 'results', 'data', 'showed',
            'significant', 'observed', 'found', 'increased', 'decreased', 'compared', 'control',
            'treatment', 'group', 'groups', 'effect', 'effects', 'level', 'levels', 'expression','within',
            # 생물학/의학 관련 불용어 추가
            'cell', 'cells', 'human', 'humans', 'patient', 'patients', 'study', 'studies',
            'analysis', 'results', 'data', 'showed', 'significant', 'observed', 'found',
            'increased', 'decreased', 'compared', 'control', 'treatment', 'group', 'groups',
            'effect', 'effects', 'level', 'levels', 'expression', 'protein', 'proteins',
            'gene', 'genes', 'response', 'activity', 'function', 'role', 'important',
            'potential', 'possible', 'previous', 'recent', 'current', 'novel', 'new',
            'research', 'investigation', 'examination', 'evaluation', 'assessment',
            'measured', 'determined', 'identified', 'demonstrated', 'revealed',
            'associated', 'related', 'induced', 'caused', 'performed', 'conducted',
            # 추가 불용어 (특허/논문에서 자주 나오는 일반적인 단어들)
            'one', 'first', 'second', 'third', 'more', 'sequence', 'alkylene', 'medium', 'tissue',
            'methods', 'culture', 'organoids', 'stem', 'bone', 'organoid', 'marrow', 'least',
            'mammalian', 'any', 'acid', 'comprises', 'culturing', 'compositions', 'crispr', 'target',
            'composition', 'producing', 'disease', 'intestinal', 'inhibitor', 'tracr', 'systems',
            'selected', 'region', 'liver', 'cancer', 'use', 'enzyme', 'matrix', 'embryoid', 'free',
            'model', 'neural', 'vascular', 'step', 'substituted', 'tumor', 'compounds', 'which',
            'ring', 'tissues', 'rna', 'drug', 'each', 'provides', 'organ', 'kinase', 'sequences',
            'nucleic', 'subject', 'independently', 'delivery', 'form', 'polynucleotide', 'vegf',
            'hours', 'derived', 'vitro', 'also', 'bodies', 'mature', 'complex', 'having',
            'combination', 'pluripotent', 'containing', 'cycloalkyl', 'produced', 'serum', 'lipid',
            'nano', 'two', 'optionally', 'surface', 'population', 'chamber', 'scf', 'mate', 'alkyl',
            'thereby', 'agent', 'dimensional', 'extracellular', 'spheroids', 'network', 'described',
            'growth', 'well', 'administering', 'claim', 'application', 'example', 'further',
            'various', 'particular', 'specific', 'multiple', 'several', 'different', 'certain',
            'example', 'preferred', 'suitable', 'effective', 'useful', 'known', 'shown'
        }
        

    @st.cache_data
    def load_data(_self, patent_file=None, paper_file=None):
        """데이터 로드 (캐시 적용)"""
        patent_df = None
        paper_df = None
        
        if patent_file is not None:
            try:
                patent_df = pd.read_csv(patent_file, encoding='utf-8')
                st.success(f"✅ 특허 데이터 로드 완료: {len(patent_df):,}건")
            except Exception as e:
                st.error(f"❌ 특허 데이터 로드 오류: {e}")
        
        if paper_file is not None:
            try:
                paper_df = pd.read_csv(paper_file, encoding='utf-8')
                st.success(f"✅ 논문 데이터 로드 완료: {len(paper_df):,}건")
            except Exception as e:
                st.error(f"❌ 논문 데이터 로드 오류: {e}")
        
        return patent_df, paper_df
    
    def preprocess_text(self, text):
        """텍스트 전처리"""
        if pd.isna(text) or text == '':
            return []
        
        text = str(text).lower()
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        words = text.split()
        
        filtered_words = []
        for word in words:
            word = word.strip()
            if (len(word) > 2 and 
                word not in self.korean_stopwords and 
                word not in self.english_stopwords and
                not word.isdigit() and
                not re.match(r'^[a-z]{1,2}$', word)):  # 1-2글자 영어 단어 제외
                filtered_words.append(word)
        
        return filtered_words
    
    def filter_by_date(self, df, date_column, start_date, end_date):
        """날짜 범위로 필터링"""
        try:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            
            mask = (df[date_column] >= start_date) & (df[date_column] <= end_date)
            return df[mask]
        except Exception as e:
            st.error(f"날짜 필터링 오류: {e}")
            return df
    
    def extract_patent_text(self, df, start_date, end_date):
        """특허 데이터에서 텍스트 추출"""
        if df is None or df.empty:
            return []
        
        df_filtered = self.filter_by_date(df, '출원일', start_date, end_date)
        all_words = []
        
        text_columns = ['발명의명칭(국문)', '발명의명칭(영문)', '요약(국문)', '요약(영문)', 
                       '대표청구항(국문)', '대표청구항(영문)']
        
        for col in text_columns:
            if col in df_filtered.columns:
                for text in df_filtered[col].dropna():
                    all_words.extend(self.preprocess_text(text))
        
        return all_words
    
    def extract_paper_text(self, df, start_date, end_date):
        """논문 데이터에서 텍스트 추출"""
        if df is None or df.empty:
            return []
        
        df_filtered = self.filter_by_date(df, '게시 날짜', start_date, end_date)
        all_words = []
        
        text_columns = ['제목', '초록', '키워드']
        
        for col in text_columns:
            if col in df_filtered.columns:
                for text in df_filtered[col].dropna():
                    all_words.extend(self.preprocess_text(text))
        
        return all_words
    
    def generate_wordcloud_data(self, patent_df, paper_df, start_date, end_date, data_type='both', top_n=100):
        """워드클라우드 데이터 생성"""
        all_words = []
        
        if data_type in ['patent', 'both']:
            patent_words = self.extract_patent_text(patent_df, start_date, end_date)
            all_words.extend(patent_words)
        
        if data_type in ['paper', 'both']:
            paper_words = self.extract_paper_text(paper_df, start_date, end_date)
            all_words.extend(paper_words)
        
        word_freq = Counter(all_words)
        top_words = dict(word_freq.most_common(top_n))
        
        return top_words, len(all_words), len(word_freq)

# 메인 앱
def main():
    st.title("📊 기간별 워드클라우드 분석기")
    st.markdown("---")
    
    # 분석기 초기화
    analyzer = StreamlitWordCloudAnalyzer()
    
    # 세션 상태 초기화
    if 'patent_df' not in st.session_state:
        st.session_state.patent_df = None
    if 'paper_df' not in st.session_state:
        st.session_state.paper_df = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    # 사이드바 설정
    st.sidebar.header("🔧 설정")
    
    # 데이터 상태 표시
    if st.session_state.data_loaded:
        st.sidebar.success("✅ 데이터가 로드되어 있습니다")
        if st.session_state.patent_df is not None:
            st.sidebar.info(f"📄 특허 데이터: {len(st.session_state.patent_df):,}건")
        if st.session_state.paper_df is not None:
            st.sidebar.info(f"📝 논문 데이터: {len(st.session_state.paper_df):,}건")
        
        # 데이터 초기화 버튼
        if st.sidebar.button("🔄 데이터 초기화", type="secondary"):
            st.session_state.patent_df = None
            st.session_state.paper_df = None
            st.session_state.data_loaded = False
            st.session_state.search_history = []
            st.rerun()
    
    # 파일 업로드 (데이터가 없을 때만 표시)
    if not st.session_state.data_loaded:
        st.sidebar.subheader("📁 데이터 업로드")
        patent_file = st.sidebar.file_uploader("특허 데이터 (CSV)", type=['csv'], key="patent")
        paper_file = st.sidebar.file_uploader("논문 데이터 (CSV)", type=['csv'], key="paper")
        
        if patent_file is not None or paper_file is not None:
            if st.sidebar.button("📥 데이터 로드", type="primary"):
                with st.spinner("데이터를 로드하는 중..."):
                    patent_df, paper_df = analyzer.load_data(patent_file, paper_file)
                    st.session_state.patent_df = patent_df
                    st.session_state.paper_df = paper_df
                    st.session_state.data_loaded = True
                    st.rerun()
    else:
        patent_df = st.session_state.patent_df
        paper_df = st.session_state.paper_df
    
    if not st.session_state.data_loaded:
        st.info("👆 사이드바에서 분석할 CSV 파일을 업로드하고 '데이터 로드' 버튼을 눌러주세요.")
        st.stop()
    
    # 데이터 로드 완료 후에는 세션 상태에서 가져오기
    patent_df = st.session_state.patent_df
    paper_df = st.session_state.paper_df
    
    # 분석 모드 선택
    st.sidebar.subheader("🎯 분석 모드")
    analysis_mode = st.sidebar.radio(
        "분석 방식 선택",
        ["단일 분석", "다중 비교 분석"],
        help="단일: 하나의 조건으로 분석 / 다중: 여러 조건을 한번에 비교"
    )
    
    if analysis_mode == "단일 분석":
        # 기존 단일 분석 설정
        st.sidebar.subheader("📊 분석 설정")
        data_type_options = []
        if patent_df is not None:
            data_type_options.append(("특허만", "patent"))
        if paper_df is not None:
            data_type_options.append(("논문만", "paper"))
        if patent_df is not None and paper_df is not None:
            data_type_options.append(("둘 다", "both"))
        
        data_type_label = st.sidebar.selectbox(
            "분석할 데이터",
            [label for label, _ in data_type_options],
            index=len(data_type_options)-1 if len(data_type_options) > 1 else 0
        )
        data_type = next(value for label, value in data_type_options if label == data_type_label)
        
        # 날짜 범위 설정
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("시작 날짜", date(2023, 1, 1))
        with col2:
            end_date = st.date_input("종료 날짜", date(2024, 12, 31))
        
        # 키워드 개수 설정
        top_n = st.sidebar.slider("표시할 키워드 수", 20, 200, 100, 10)
        
        # 단일 분석 실행
        if st.sidebar.button("🚀 분석 실행", type="primary"):
            # 검색 히스토리에 추가
            search_config = {
                "data_type": data_type_label,
                "start_date": str(start_date),
                "end_date": str(end_date),
                "top_n": top_n,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 중복 검색 방지 (같은 설정이면 추가하지 않음)
            if not any(h['data_type'] == search_config['data_type'] and 
                      h['start_date'] == search_config['start_date'] and 
                      h['end_date'] == search_config['end_date'] 
                      for h in st.session_state.search_history):
                st.session_state.search_history.append(search_config)
                # 최대 10개까지만 저장
                if len(st.session_state.search_history) > 10:
                    st.session_state.search_history.pop(0)
            
            # 단일 분석 실행
            execute_single_analysis(analyzer, patent_df, paper_df, data_type, data_type_label, start_date, end_date, top_n)
    
    else:  # 다중 비교 분석
        st.sidebar.subheader("📊 다중 비교 설정")
        
        # 세션 상태에 비교 설정 저장
        if 'comparison_configs' not in st.session_state:
            st.session_state.comparison_configs = []
        
        # 새로운 비교 조건 추가
        with st.sidebar.expander("➕ 새 조건 추가", expanded=True):
            # 데이터 유형 선택
            data_type_options = []
            if patent_df is not None:
                data_type_options.append(("특허만", "patent"))
            if paper_df is not None:
                data_type_options.append(("논문만", "paper"))
            if patent_df is not None and paper_df is not None:
                data_type_options.append(("둘 다", "both"))
            
            new_data_type_label = st.selectbox(
                "데이터 유형",
                [label for label, _ in data_type_options],
                key="new_data_type"
            )
            new_data_type = next(value for label, value in data_type_options if label == new_data_type_label)
            
            # 날짜 범위
            col1, col2 = st.columns(2)
            with col1:
                new_start_date = st.date_input("시작", date(2023, 1, 1), key="new_start")
            with col2:
                new_end_date = st.date_input("종료", date(2024, 12, 31), key="new_end")
            
            # 조건 이름
            condition_name = st.text_input("조건 이름", f"{new_data_type_label}({new_start_date}~{new_end_date})", key="new_name")
            
            # 추가 버튼
            if st.button("➕ 조건 추가"):
                new_config = {
                    "name": condition_name,
                    "data_type": new_data_type,
                    "data_type_label": new_data_type_label,
                    "start_date": new_start_date,
                    "end_date": new_end_date
                }
                st.session_state.comparison_configs.append(new_config)
                st.rerun()
        
        # 추가된 조건들 표시
        if st.session_state.comparison_configs:
            st.sidebar.subheader("📋 비교 조건 목록")
            for i, config in enumerate(st.session_state.comparison_configs):
                col1, col2 = st.sidebar.columns([3, 1])
                with col1:
                    st.write(f"**{config['name']}**")
                    st.caption(f"{config['data_type_label']} | {config['start_date']} ~ {config['end_date']}")
                with col2:
                    if st.button("❌", key=f"remove_{i}"):
                        st.session_state.comparison_configs.pop(i)
                        st.rerun()
            
            # 전체 키워드 수 설정
            comparison_top_n = st.sidebar.slider("각 조건별 키워드 수", 20, 100, 50, 10)
            
            # 다중 비교 실행
            if st.sidebar.button("🔍 다중 비교 실행", type="primary"):
                execute_comparison_analysis(analyzer, patent_df, paper_df, st.session_state.comparison_configs, comparison_top_n)
            
            # 조건 전체 삭제
            if st.sidebar.button("🗑️ 모든 조건 삭제"):
                st.session_state.comparison_configs = []
                st.rerun()
# 단일 분석 실행 함수
def execute_single_analysis(analyzer, patent_df, paper_df, data_type, data_type_label, start_date, end_date, top_n):
    with st.spinner("워드클라우드를 생성하는 중..."):
        # 워드클라우드 데이터 생성
        word_freq, total_words, unique_words = analyzer.generate_wordcloud_data(
            patent_df, paper_df, start_date, end_date, data_type, top_n
        )
        
        if not word_freq:
            st.warning("⚠️ 선택한 기간과 조건에 해당하는 데이터가 없습니다.")
            st.stop()
        
        # 통계 정보 표시
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📝 총 단어 수", f"{total_words:,}")
        with col2:
            st.metric("🔤 고유 단어 수", f"{unique_words:,}")
        with col3:
            st.metric("⭐ 상위 키워드", f"{len(word_freq)}")
        with col4:
            st.metric("📊 최고 빈도", f"{max(word_freq.values()) if word_freq else 0}")
        
        st.markdown("---")
        
        # 탭으로 결과 표시
        tab1, tab2, tab3 = st.tabs(["🌟 워드클라우드", "📈 키워드 순위", "📋 상세 목록"])
        
        with tab1:
            st.subheader("🌟 워드클라우드")
            
            # 워드클라우드 생성
            fig, ax = plt.subplots(figsize=(15, 8))
            
            try:
                wordcloud = WordCloud(
                    width=1200, 
                    height=600,
                    background_color='white',
                    max_words=top_n,
                    colormap='viridis',
                    random_state=42,
                    font_path=None  # 시스템 기본 폰트 사용
                ).generate_from_frequencies(word_freq)
                
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title(f'워드클라우드: {data_type_label} ({start_date} ~ {end_date})', 
                            fontsize=16, fontweight='bold', pad=20)
                
                st.pyplot(fig)
                
                # 워드클라우드 이미지 다운로드 버튼
                import io
                img_buffer = io.BytesIO()
                fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                img_buffer.seek(0)
                
                st.download_button(
                    label="📥 워드클라우드 이미지 다운로드 (PNG)",
                    data=img_buffer.getvalue(),
                    file_name=f"wordcloud_{data_type}_{start_date}_{end_date}.png",
                    mime="image/png"
                )
                
            except Exception as e:
                st.error(f"워드클라우드 생성 오류: {e}")
                st.info("💡 한글 폰트가 없어서 오류가 발생할 수 있습니다.")
        
        with tab2:
            st.subheader("📈 상위 키워드 순위")
            
            # 상위 30개 키워드 막대 그래프
            top_30 = dict(list(word_freq.items())[:30])
            
            fig = go.Figure([go.Bar(
                x=list(top_30.values()),
                y=list(top_30.keys()),
                orientation='h',
                marker_color='lightblue',
                text=list(top_30.values()),
                textposition='auto',
            )])
            
            fig.update_layout(
                title=f"상위 30개 키워드 빈도 ({start_date} ~ {end_date})",
                xaxis_title="빈도",
                yaxis_title="키워드",
                height=800,
                yaxis={'categoryorder':'total ascending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("📋 키워드 상세 목록")
            
            # 데이터프레임으로 표시
            df_keywords = pd.DataFrame([
                {"순위": i+1, "키워드": word, "빈도": count, "비율(%)": round(count/total_words*100, 2)}
                for i, (word, count) in enumerate(word_freq.items())
            ])
            
            # 검색 기능
            search_term = st.text_input("🔍 키워드 검색")
            if search_term:
                df_filtered = df_keywords[df_keywords['키워드'].str.contains(search_term, case=False, na=False)]
                st.dataframe(df_filtered, use_container_width=True)
            else:
                st.dataframe(df_keywords, use_container_width=True)
            
            # CSV 다운로드 버튼
            csv = df_keywords.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 키워드 목록 다운로드 (CSV)",
                data=csv,
                file_name=f"keywords_{data_type}_{start_date}_{end_date}.csv",
                mime="text/csv"
            )

# 다중 비교 분석 실행 함수
def execute_comparison_analysis(analyzer, patent_df, paper_df, configs, top_n):
    if len(configs) < 2:
        st.warning("⚠️ 비교 분석을 위해서는 최소 2개 이상의 조건이 필요합니다.")
        return
    
    with st.spinner("다중 비교 분석을 실행하는 중..."):
        comparison_results = {}
        
        # 각 조건별로 워드클라우드 데이터 생성
        for config in configs:
            word_freq, total_words, unique_words = analyzer.generate_wordcloud_data(
                patent_df, paper_df, 
                config['start_date'], config['end_date'], 
                config['data_type'], top_n
            )
            
            comparison_results[config['name']] = {
                'word_freq': word_freq,
                'total_words': total_words,
                'unique_words': unique_words,
                'config': config
            }
        
        # 결과 표시
        st.subheader("🔍 다중 조건 비교 분석 결과")
        
        # 전체 통계 비교
        st.subheader("📊 전체 통계 비교")
        stats_data = []
        for name, result in comparison_results.items():
            stats_data.append({
                "조건": name,
                "총 단어 수": result['total_words'],
                "고유 단어 수": result['unique_words'],
                "상위 키워드 수": len(result['word_freq']),
                "최고 빈도": max(result['word_freq'].values()) if result['word_freq'] else 0
            })
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
        
        # 워드클라우드 그리드 표시
        st.subheader("🌟 워드클라우드 비교")
        
        # 2x2 또는 1xN 그리드로 표시
        n_configs = len(configs)
        if n_configs <= 4:
            cols = 2
            rows = (n_configs + 1) // 2
        else:
            cols = 3
            rows = (n_configs + 2) // 3
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 6*rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, (name, result) in enumerate(comparison_results.items()):
            if i >= len(axes):
                break
                
            try:
                if result['word_freq']:
                    wordcloud = WordCloud(
                        width=600, 
                        height=400,
                        background_color='white',
                        max_words=top_n//2,  # 각 워드클라우드는 절반 크기로
                        colormap='viridis',
                        random_state=42,
                        font_path=None
                    ).generate_from_frequencies(result['word_freq'])
                    
                    axes[i].imshow(wordcloud, interpolation='bilinear')
                    axes[i].set_title(name, fontsize=12, fontweight='bold')
                    axes[i].axis('off')
                else:
                    axes[i].text(0.5, 0.5, '데이터 없음', ha='center', va='center', fontsize=14)
                    axes[i].set_title(name, fontsize=12, fontweight='bold')
                    axes[i].axis('off')
            except Exception as e:
                axes[i].text(0.5, 0.5, f'오류: {str(e)}', ha='center', va='center', fontsize=10)
                axes[i].set_title(name, fontsize=12, fontweight='bold')
                axes[i].axis('off')
        
        # 빈 subplot 숨기기
        for i in range(len(comparison_results), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # 키워드 교집합 분석
        st.subheader("🔗 공통 키워드 분석")
        
        # 모든 조건의 키워드 수집
        all_keywords = {}
        for name, result in comparison_results.items():
            for word, freq in result['word_freq'].items():
                if word not in all_keywords:
                    all_keywords[word] = {}
                all_keywords[word][name] = freq
        
        # 공통 키워드 찾기 (2개 이상 조건에서 나타나는 키워드)
        common_keywords = {word: data for word, data in all_keywords.items() 
                          if len(data) >= 2}
        
        if common_keywords:
            # 공통 키워드 표 생성
            common_df_data = []
            for word, conditions in common_keywords.items():
                row = {"키워드": word}
                total_freq = 0
                for config in configs:
                    freq = conditions.get(config['name'], 0)
                    row[config['name']] = freq
                    total_freq += freq
                row["전체 빈도"] = total_freq
                common_df_data.append(row)
            
            # 전체 빈도순으로 정렬
            common_df = pd.DataFrame(common_df_data).sort_values("전체 빈도", ascending=False)
            
            st.write(f"**2개 이상 조건에서 나타나는 공통 키워드: {len(common_keywords)}개**")
            st.dataframe(common_df.head(20), use_container_width=True)
            
            # 공통 키워드 CSV 다운로드
            csv_common = common_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 공통 키워드 분석 결과 다운로드 (CSV)",
                data=csv_common,
                file_name="common_keywords_analysis.csv",
                mime="text/csv"
            )
        else:
            st.info("공통으로 나타나는 키워드가 없습니다.")

        # 워드클라우드 비교 이미지 다운로드
        import io
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        img_buffer.seek(0)
        
        st.download_button(
            label="📥 비교 워드클라우드 이미지 다운로드 (PNG)",
            data=img_buffer.getvalue(),
            file_name="comparison_wordclouds.png",
            mime="image/png"
        )

    # 검색 히스토리 표시
    if st.session_state.search_history:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📚 검색 히스토리")
        
        for i, history in enumerate(reversed(st.session_state.search_history[-5:])):  # 최근 5개만 표시
            with st.sidebar.expander(f"🔍 {history['data_type']} ({history['start_date']})"):
                st.write(f"**데이터 유형:** {history['data_type']}")
                st.write(f"**기간:** {history['start_date']} ~ {history['end_date']}")
                st.write(f"**키워드 수:** {history['top_n']}")
                st.write(f"**검색 시간:** {history['timestamp']}")
                
                # 재실행 버튼
                if st.button(f"🔄 재실행", key=f"rerun_{i}"):
                    # 설정 복원
                    st.session_state.temp_data_type = history['data_type']
                    st.session_state.temp_start_date = history['start_date']
                    st.session_state.temp_end_date = history['end_date']
                    st.session_state.temp_top_n = history['top_n']
                    st.rerun()
        
        # 히스토리 전체 삭제 버튼
        if st.sidebar.button("🗑️ 히스토리 삭제"):
            st.session_state.search_history = []
            st.rerun()
    
    # 사용법 안내
    st.sidebar.markdown("---")
    st.sidebar.subheader("📖 사용법")
    st.sidebar.markdown("""
    1. **데이터 업로드**: 특허 또는 논문 CSV 파일 업로드
    2. **데이터 로드**: '데이터 로드' 버튼 클릭 (한 번만!)
    3. **분석 설정**: 데이터 유형과 기간 선택
    4. **분석 실행**: 버튼을 눌러 워드클라우드 생성
    5. **히스토리**: 이전 검색 결과를 빠르게 재실행
    """)
    
    # 추가 정보
    with st.expander("ℹ️ 데이터 형식 안내"):
        st.markdown("""
        **특허 데이터 컬럼:**
        - 출원일, 발명의명칭(국문/영문), 요약(국문/영문), 대표청구항(국문/영문)
        
        **논문 데이터 컬럼:**
        - 게시 날짜, 제목, 초록, 키워드
        
        **불용어 처리:**
        - 한국어: 조사, 어미, 일반적인 단어
        - 영어: 관사, 전치사, 일반적인 연구 용어 (cell, human, study 등)
        """)

if __name__ == "__main__":
    main()