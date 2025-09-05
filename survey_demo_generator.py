"""
Survey Demo Data Generator
Single-file engine + Streamlit UI for generating realistic survey demo datasets.

How it works (short):
- Infer question type from text (heuristic) or accept user override.
- Build QuestionSpec records with generation parameters (categories, probs, scale, etc.).
- Use a Gaussian copula to generate correlated latent variables for questions that support correlation (ordinal, numeric, ranking, text via sentiment scores).
- Map latent variables to categorical/ordinal outputs via thresholds.
- Independently sample pure categorical or multi-label questions.

Run:
    pip install pandas numpy scipy streamlit
    streamlit run survey_demo_generator.py

This file contains:
- Question type inference heuristics
- Generator engine (generate_responses)
- Simple Streamlit interactive UI to override types, set params, and build pairwise correlations

Feel free to ask for custom tweaks: richer text templates, different copula families, multi-label correlation, or an export template for your pipeline.
"""
import json
import numpy as np
import pandas as pd


from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from scipy.stats import norm, multivariate_normal

# # Optional UI
# try:
#     import streamlit as st
# except Exception:
#     st = None
import streamlit as st

# Global variables
SUPPORTED_QUESTION_TYPES = ['ordinal', 'categorical', 'numeric', 'ranking', 'text', 'multi-label', 'nps']
SUPPORTED_DIST_TYPES = ['uniform', 'poisson', 'normal', 'scaled_beta']

# Heuristics: infer question type
ORDINAL_KEYWORDS = ['rate', 'rating', 'how satisfied', 'how likely', 'scale', 'rank (1', 'on a scale', 'score']
RANKING_KEYWORDS = ['rank', 'order the following', 'which of the following do you prefer']
MULTI_KEYWORDS = ['select all', 'check all', 'choose any', 'select any']
NUMERIC_KEYWORDS = ['how many', 'how much', 'number of', 'years of']
TEXT_KEYWORDS = ['please explain', 'why', 'what', 'other', 'please describe', 'comments']

#Preface of the streamlit app
ST_APP_HEADER = '''
Data generator can:
- Accept a JSON list of questions or a pasted newline-separated list.
- Infer question types and let you override via dropdowns.
- Let you set pairwise correlations across questions.
- Generate sample dataset and download as CSV.
'''


@dataclass
class QuestionSpec:
    id: str
    text: str
    inferred_type: str  # one of SUPPORTED_QUESTION_TYPES
    options: Optional[List[str]] = None  # for categorical / ranking / ordinal
    k: Optional[int] = None  # number of categories for ordinal
    probs: Optional[List[float]] = None  # category probabilities
    scale_min: Optional[float] = None
    scale_max: Optional[float] = None
    distribution: Optional[str] = 'uniform'  # or 'poisson', 'normal', 'skewed', 'scaled_beta', 'uniform'
    distribution_params = None
    default_val = None
    # For text fields we keep simple templates
    text_templates: Optional[Dict[str, List[str]]] = None


def infer_question_type(text: str, options: Optional[List[str]] = None) -> str:
    t = text.lower()
    if options is not None and len(options) > 0:
        # If options are ordered phrases like '1 - Strongly disagree' detect ordinal
        joined = ' '.join(options).lower()
        if any(k in joined for k in ['strongly agree', 'strongly disagree', 'agree', 'disagree']):
            return 'ordinal'
        if any(k in t for k in RANKING_KEYWORDS) or any('rank' in o.lower() for o in options[:5]):
            return 'ranking'
        if any(k in t for k in MULTI_KEYWORDS) or any('select all' in o.lower() for o in options):
            return 'multi-label'
        # otherwise categorical
        return 'categorical'

    if any(k in t for k in RANKING_KEYWORDS):
        return 'ranking'
    if any(k in t for k in ORDINAL_KEYWORDS):
        return 'ordinal'
    if any(k in t for k in NUMERIC_KEYWORDS):
        return 'numeric'
    if any(k in t for k in MULTI_KEYWORDS):
        return 'multi-label'
    if any(k in t for k in TEXT_KEYWORDS):
        return 'text'
    # fallback: categorical
    return 'categorical'


def default_probs(k: int) -> List[float]:
    p = np.ones(k) / k
    return p.tolist()


def make_default_spec(qid: int, text: str, options: Optional[List[str]] = None) -> QuestionSpec:
    qtype = infer_question_type(text, options)
    if qtype == 'ordinal':
        k = 5
        probs = default_probs(k)
        return QuestionSpec(id=str(qid), text=text, inferred_type='ordinal', k=k, probs=probs,
                            scale_min=1, scale_max=5)
    if qtype == 'ranking':
        k = len(options) if options else 4
        probs = default_probs(k)
        opts = options if options else [f'Item {i+1}' for i in range(k)]
        return QuestionSpec(id=str(qid), text=text, inferred_type='ranking', options=opts, k=k, probs=probs)
    if qtype == 'multi-label':
        opts = options if options else [f'Option {i+1}' for i in range(4)]
        return QuestionSpec(id=str(qid), text=text, inferred_type='multi-label', options=opts, probs=None)
    if qtype == 'numeric':
        return QuestionSpec(id=str(qid), text=text, inferred_type='numeric', scale_min=0, scale_max=100)
    if qtype == 'text':
        templates = {
            'positive': ["I really liked the product.", "Exceeded expectations."],
            'neutral': ["It was okay.", "Not bad overall."],
            'negative': ["I was disappointed.", "Needs improvement."]
        }
        return QuestionSpec(id=str(qid), text=text, inferred_type='text', text_templates=templates)

    # categorical fallback
    opts = options if options else [f'Choice {i+1}' for i in range(4)]
    probs = default_probs(len(opts))
    return QuestionSpec(id=str(qid), text=text, inferred_type='categorical', options=opts, probs=probs)


def build_covariance_matrix(specs: List[QuestionSpec], pairwise_corr: Dict[Tuple[int,int], float]) -> np.ndarray:
    # Build a covariance (correlation) matrix for the latent Gaussian variables.
    n = len(specs)
    corr = np.eye(n)
    for (i,j), v in pairwise_corr.items():
        corr[i, j] = v
        corr[j, i] = v
    # ensure positive semi-definite: small eigenvalue correction
    eps = 1e-6
    try:
        # if not psd, adjust
        eigs, vecs = np.linalg.eigh(corr)
        if np.any(eigs < 0):
            eigs[eigs < 0] = eps
            corr = vecs @ np.diag(eigs) @ vecs.T
            # rescale diagonal to 1
            d = np.sqrt(np.diag(corr))
            corr = corr / d[:, None] / d[None, :]
    except Exception:
        corr = np.eye(n)
    return corr


def map_latent_to_ordinal(latent: np.ndarray, probs: List[float]) -> np.ndarray:
    # latent: shape (n_samples, ) - standard normal
    # probs: list of k probs
    cum = np.cumsum(probs)
    thresholds = norm.ppf(cum[:-1])  # length k-1
    # assign category 0..k-1
    cats = np.searchsorted(thresholds, latent)
    return cats


def standardize_probs(spec):
    if spec.probs is None:
        if spec.k is not None:
            return default_probs(len(spec.k))
        else:
            return default_probs(len(spec.options))
    else:
        total = sum(spec.probs)
        return [freq/total for freq in spec.probs]


def generate_nps_probs(nps, mnp=.6):
    adj_nps = nps/100
    perc_neut = mnp*(1-abs(adj_nps)) # mnp: max neutral percentage, .6 is empirically fairly accurate
    perc_prom = (1-perc_neut + adj_nps)/2
    perc_detr = perc_prom-adj_nps

    perc10 = perc_prom*(.55+.19*adj_nps) # Adjusts based on NPS, high nps means more 10's
    perc9 = perc_prom - perc10

    perc8 = perc_prom * (.50 + .12 * adj_nps) # Same logic as for 10's
    perc7 = perc_prom - perc8

    perc6 = .269*perc_detr  # Doesn't adjust much given an NPS
    perc5 = .353*perc_detr
    perc4 = .098*perc_detr
    perc3 = .078*perc_detr
    perc2 = .060*perc_detr
    perc1 = .056*perc_detr
    perc0 = .086*perc_detr

    return [perc0, perc1, perc2, perc3, perc4, perc5, perc6, perc7, perc8, perc9, perc10]


def generate_responses(specs: List[QuestionSpec], n: int = 500,
                       pairwise_corr: Optional[Dict[Tuple[int,int], float]] = None,
                       random_state: Optional[int] = None,
                       decay_percent: Optional[float] = 0) -> pd.DataFrame:
    rng = np.random.RandomState(random_state)
    m = len(specs)
    pairwise_corr = pairwise_corr or {}

    # Prepare which questions will share latent variables (ordinal, numeric, ranking, text)
    correlated_indices = [i for i,s in enumerate(specs) if s.inferred_type in ('ordinal','numeric','ranking','text')]
    # For simplicity: we'll build a latent vector for ALL specs, but categorical/multi-label will ignore correlation
    corr_matrix = build_covariance_matrix(specs, pairwise_corr)

    # Sample latent multivariate normal
    latent = rng.multivariate_normal(mean=np.zeros(m), cov=corr_matrix, size=n)

    rows = []
    for r in range(n):
        row = {}
        decay_chance = 0
        for i,s in enumerate(specs):
            if decay_chance < decay_percent:
                break
            lv = latent[r, i]
            if s.inferred_type == 'ordinal':
                probs = standardize_probs(s) if s.probs is not None else default_probs(len(s.options))
                cat = int(map_latent_to_ordinal(np.array([lv]), probs)[0])
                # map to scale_min..scale_max
                val = int(round((s.scale_min if s.scale_min is not None else 1) + cat))
                row[s.id] = val
            elif s.inferred_type == 'nps':
                probs = standardize_probs(s) if s.probs is not None else default_probs(len(s.options))
                val = int(map_latent_to_ordinal(np.array([lv]), probs)[0])
                row[s.id] = val
            elif s.inferred_type == 'numeric':
                # map latent to numeric range via probability density
                u = norm.cdf(lv)
                lo = s.scale_min if s.scale_min is not None else 0
                hi = s.scale_max if s.scale_max is not None else 100
                val = round(min(max(lo, u), hi), 0) # Don't round in the future if a float is required
                row[s.id] = float(val)
            elif s.inferred_type == 'categorical':
                probs = standardize_probs(s) if s.probs is not None else default_probs(len(s.options))
                choice = rng.choice(s.options, p=probs)
                row[s.id] = choice
            # elif s.inferred_type == 'ranking':
            #     # For ranking, assume options exist. We'll sample small set of latent scores for each option
            #     # We'll use the same latent value as a bias and add small noise per option
            #     base = lv
            #     k = len(s.options)
            #     option_scores = base + rng.normal(scale=1.0, size=k)
            #     option_scores = np.array([score for score in option_scores if score < 1.5])
            #     rank_idx = np.argsort(-option_scores)  # highest first
            #     # store as semi-colon-separated ranking
            #     ranked = [s.options[j] for j in rank_idx]
            #     row[s.id] = ';'.join(ranked)
            elif s.inferred_type in ('multi-label', 'ranking'):
                probs = s.probs if s.probs is not None else default_probs(len(s.options))
                chosen = [opt for opt,p in zip(s.options, probs) if rng.rand() < p]
                row[s.id] = ';'.join(chosen)
            elif s.inferred_type == 'text':
                # map latent to sentiment band
                u = norm.cdf(lv)
                if u > 0.66:
                    band = 'positive'
                elif u > 0.33:
                    band = 'neutral'
                else:
                    band = 'negative'
                templates = s.text_templates if s.text_templates else {
                    'positive': ['Loved it.'], 'neutral': ['It was ok.'], 'negative': ['Disappointed.']
                }
                txt = rng.choice(templates[band])
                row[s.id] = txt
            else:
                row[s.id] = None
            decay_chance = rng.rand()
        rows.append(row)

    df = pd.DataFrame(rows)
    # attach human-friendly column names as question text
    rename = {s.id: f"Q{s.id}: {s.text[:60]}" for s in specs}
    df.rename(columns=rename, inplace=True)
    return df


# -------------------------
# Small helper: build specs from question list
# -------------------------

def build_specs_from_questions(questions: List[Dict[str,Any]]) -> List[QuestionSpec]:
    specs = []
    for i,q in enumerate(questions):
        text = q.get('text') or q.get('question') or f'Question {i+1}'
        options = q.get('options')
        spec = make_default_spec(i+1, text, options)
        # allow requests to override: q may contain explicit params
        if 'type' in q:
            spec.inferred_type = q['type']
        if 'probs' in q:
            spec.probs = q['probs']

        if 'options' in q:
            spec.options = q['options']
        if 'scale_min' in q:
            spec.scale_min = q['scale_min']
        if 'scale_max' in q:
            spec.scale_max = q['scale_max']
        specs.append(spec)
    return specs

def pick_questions():
    example_questions = [
        {"text": "How satisfied are you with our service?"},
        {"text": "Please rank the following features in order of importance:",
         "options": ["Price", "Reliability", "UX", "Support"]},
        {"text": "Which of these benefits did you use? (select all that apply)",
         "options": ["Benefit A", "Benefit B", "Benefit C"]},
        {"text": "Any other comments?"}
    ]

    qinput_mode = st.radio('Provide questions as', ['Example', 'Paste JSON', 'Paste newline list'])
    questions = None
    if qinput_mode == 'Example':
        st.write('Using built-in example questions')
        questions = example_questions
    elif qinput_mode == 'Paste JSON':
        raw = st.text_area('Paste questions as JSON list', height=200)
        if raw.strip():
            try:
                questions = json.loads(raw)
            except Exception as e:
                st.error(f'JSON parse error: {e}')
    else:
        raw = st.text_area('Paste one question per line', value='\n'.join([q['text'] for q in example_questions]))
        if raw.strip():
            questions = [{'text': line.strip()} for line in raw.strip().splitlines() if line.strip()]

    if questions is None:
        st.stop()

    return questions


def display_question_name(column, i, text):
    column.write(f"**Q{i + 1}.** {text}")


def get_question_type(col, i, inferred_type):
    return col.selectbox('Type', SUPPORTED_QUESTION_TYPES, index=SUPPORTED_QUESTION_TYPES.index(inferred_type), key=f'type_{i}')


def customize_questions(specs):
    st.subheader('Detected question types (override if wrong)')
    for i,s in enumerate(specs):
        st.divider()
        col1, col2, col3, col4 = st.columns([4,2,4,2])

        display_question_name(col1, i, s.text)

        s.inferred_type = get_question_type(col2, i, s.inferred_type)

        if s.inferred_type in ('categorical','ranking','multi-label'):

            # show options editor
            opts = col3.text_input('Options (semi-colon-separated)', value=';'.join(s.options) if s.options else 'Option1;Option2;Option3', key=f'opts_{i}')
            s.options = [o.strip() for o in opts.split(';') if o.strip()]
            prob_cols = st.columns(len(s.options))
            freqs = []
            for j, col in enumerate(prob_cols):
                freqs.append(col.number_input(f"{s.options[j]} probability.", value=1 / len(s.options), min_value=0.001, max_value=.999))
            s.probs = freqs
            s.default_val = col4.text_input('Default/Other Val', value=None, key=f'def_val_{i}') # Not sure what adding '' does...
            # if s.inferred_type == 'categorical' and s.probs is None:
            #     s.probs = default_probs(len(s.options))

        if s.inferred_type == 'ordinal':
            # set k
            k = col3.number_input('Scale (k)', min_value=2, max_value=10, value=s.k if s.k else 5)
            s.k = int(k)
            s.scale_min = 1
            s.scale_max = int(k)
            if s.probs is None:
                s.probs = default_probs(s.k)
            s.distribution = col4.selectbox('Dist', SUPPORTED_DIST_TYPES, index=SUPPORTED_DIST_TYPES.index(s.distribution),
                                      key=f'dist_{i}')
        if s.inferred_type == 'numeric':
            with col3:
                sub_col1, sub_col2 = st.columns(2)
                min_val = sub_col1.number_input("Minimum value", value=1)
                max_val = sub_col2.number_input("Maximum value", value=5)
            s.scale_min = min_val
            s.scale_max = max_val
            s.k = max_val - min_val + 1
            s.distribution = col4.selectbox('Dist', SUPPORTED_DIST_TYPES, index=SUPPORTED_DIST_TYPES.index(s.distribution),
                                      key=f'dist_{i}')

            if s.probs is None:
                s.probs = default_probs(s.k)

        if s.inferred_type == 'nps':
            nps = col3.number_input('NPS score (-100, 100)', min_value=-100, max_value=100, value=33)
            s.scale_min = 0
            s.scale_max = 10
            s.k = 11
            s.probs = generate_nps_probs(nps)

        if s.inferred_type == 'text':
            opts = col3.text_input('Options (semi-colon-separated)',
                                   value=';'.join(s.options) if s.options else 'Option1;Option2;Option3',
                                   key=f'opts_{i}')
            s.options = [o.strip() for o in opts.split(';') if o.strip()]
            prob_cols = st.columns(len(s.options))
            freqs = []
            for i, col in enumerate(prob_cols):
                freqs.append(col.number_input(f"{s.options[i]} probability.", value=1 / len(s.options), min_value=0.001,
                                              max_value=.999))
            s.probs = freqs


def pick_correlations(specs):
    st.subheader('Pairwise correlations (only affects ordinal/numeric/ranking/text)')
    st.markdown('Choose pairs to set a correlation (-0.9 to 0.9). Leave blank for 0.')
    pairwise_corr = {}
    pairs_to_set = st.multiselect('Choose question pairs to correlate',
                                  options=[(i, j) for i in range(len(specs)) if specs[i].inferred_type in ("ordinal", "numeric", "nps", "text")
                                           for j in range(i + 1, len(specs)) if specs[j].inferred_type in ("ordinal", "numeric", "nps", "text")],
                                  format_func=lambda x: f'Q{x[0] + 1} - Q{x[1] + 1}')
    for pair in pairs_to_set:
        k = f'corr_{pair[0]}_{pair[1]}'
        v = st.slider(f'Correlation Q{pair[0] + 1} vs Q{pair[1] + 1}', min_value=-0.9, max_value=0.9, value=0.4,
                      step=0.05, key=k)
        pairwise_corr[(pair[0], pair[1])] = v

    return pairwise_corr

def data_generation_options():
    st.subheader('Data Generation specs')
    n = st.number_input('Number of respondents', min_value=10, max_value=20000, value=500)
    seed = st.number_input('Random seed (0 for random)', min_value=0, max_value=999999, value=0)
    if seed == 0:
        seed = None
    decay = st.toggle("Survey Decay over time")
    decay_percent = 0
    if decay:
        decay_percent = st.number_input('Decay percent per question', min_value=0, max_value=.99, value=0.05)
    return n, seed, decay_percent

def generate_dataset(specs, pairwise_corr, n = 100, seed = None, decay_percent = 0):
    if st.button('Generate dataset'):
        df = generate_responses(specs, n=n, pairwise_corr=pairwise_corr, random_state=seed, decay_percent=decay_percent)
        st.success('Generated dataset')
        st.dataframe(df.head(100))
        csv = df.to_csv(index=False)
        st.download_button('Download CSV', data=csv, file_name='survey_demo_data.csv')


def run_streamlit_app():
    st.set_page_config(
        page_title="My App",
        page_icon="📊",
        layout="wide",  # fills the page width
        initial_sidebar_state="expanded",  # or "collapsed"
    )

    st.title('Survey Demo Data Generator (basic)')
    st.markdown(ST_APP_HEADER)

    questions = pick_questions()
    specs = build_specs_from_questions(questions)
    customize_questions(specs)
    pairwise_corr = pick_correlations(specs)
    n, seed, decay_percent = data_generation_options()
    generate_dataset(specs, pairwise_corr, n, seed, decay_percent)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Survey demo data generator (script mode)')
    parser.add_argument('--questions-file', type=str, help='JSON file with questions list')
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.questions_file:
        with open(args.questions_file, 'r') as f:
            questions = json.load(f)
        specs = build_specs_from_questions(questions)
        df = generate_responses(specs, n=args.n, random_state=(args.seed or None))
        print(df.head().to_csv(index=False))
    else:
        if st is None:
            print('No questions file provided and Streamlit not available. Run with --questions-file or install streamlit to use the UI.')
        elif st is None:
            raise RuntimeError('Streamlit is not installed.')
        else:
            run_streamlit_app()

    ## For debugging
    # n = 10
    # example_questions = [
    #     {"text": "How satisfied are you with our service?"},
    #     {"text": "Please rank the following features in order of importance:",
    #      "options": ["Price", "Reliability", "UX", "Support"]},
    #     {"text": "Which of these benefits did you use? (select all that apply)",
    #      "options": ["Benefit A", "Benefit B", "Benefit C"]},
    #     {"text": "Any other comments?"}
    # ]
    #
    # specs = build_specs_from_questions(example_questions)
    # df = generate_responses(specs, n=n, random_state=None)
