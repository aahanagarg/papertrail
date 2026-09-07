Project 2 — PaperTrail: RAG question-answering over research papers, and a reliability benchmark

Finalised 2026-09-07. Supersedes the earlier FinQA plan (she rejected finance). Owner: Aahana Garg, UMass CS '28 (junior). Target: summer 2027 AI/ML internships.

Constraints that shaped this
She wants LLM and AWS keywords. 100 applications → 0 interviews last cycle; a peer with unremarkable LLM projects got interviews. Keyword screens, not quality screens.
No AWS credits available (UMass doesn't provide them). So: train free, deploy cheap.
No finance domain.
Real users deferred — she'll pursue that later alongside research. Noted as the single biggest remaining gap vs peer resumes (see claude/peer-resume-comparison notes in conversation: peers have "adopted by UMass Chemistry dept", "600+ users", research positions — those lines outperform any benchmark number).
What it is, in one sentence

A question-answering system over research papers that shows which paragraph each answer came from — plus a rigorous measurement of how often it makes things up and whether its failures come from retrieval or from generation.

The tool is the apparatus. The measurement is the contribution. Same shape as the art detector: nobody needed another detector; the value was proving it collapses on an unseen generator.

Why QASPER

allenai/qasper on HuggingFace — questions about NLP papers, each answer annotated with the exact evidence paragraphs it should come from, plus a set of unanswerable questions. That labelled evidence is what makes the evaluation real rather than eyeballed: retrieval and generation can be scored separately against ground truth, and the hallucination rate on unanswerable questions is directly measurable.

Stack (all free or free-tier)
Layer	Choice	Cost
Corpus + ground truth	QASPER (HuggingFace datasets)	free
Sparse baseline	BM25 (rank_bm25)	free
Embeddings	sentence-transformers/all-MiniLM-L6-v2	free, local
Vector store	FAISS (local), optionally pgvector later	free
Generator	Qwen2.5-1.5B-Instruct; one API model as a ceiling comparison	~$5
Fine-tuning	LoRA via peft on Kaggle free GPU (T4/P100, 30 hrs/week)	free
Serving	FastAPI + Docker	free
Deployment	AWS free tier — Lambda container image or EC2 t3.micro; S3 for artifacts; ECR for the image	~$0
CI	GitHub Actions (free for public repos)	free
Monitoring	CloudWatch logs + latency/cost per query	free tier

Key insight driving this: cloud training is expensive, cloud serving is nearly free. Train on Kaggle, deploy on AWS. "Deployed on AWS" is honestly earned either way.

The four measurements (the actual contribution)
Retrieval recall@k — did the retriever surface the annotated evidence paragraph? Compare BM25 vs dense embeddings vs hybrid.
Answer quality — QASPER's official answer-F1 against the reference answers.
Hallucination rate on unanswerable questions — QASPER marks questions the paper does not answer. Does the system say "not stated", or invent something? This is the headline number and it is what companies deploying RAG actually want to know.
Failure decomposition — when the answer is wrong, was the evidence retrieved or not? Splits errors into retrieval failures vs generation failures. Almost nobody reports this and it is the first thing anyone debugging RAG needs.

Plus the transfer from project 1: is it confidently wrong? Log answer-token logprobs, compute ECE, check whether confidence separates right from wrong answers.

Fine-tuning arm

LoRA-fine-tune Qwen2.5-1.5B on QASPER train, on Kaggle. Then compare four configurations on the same test set:

base model, no retrieval
base model + retrieval (RAG)
fine-tuned, no retrieval
fine-tuned + retrieval

"Does fine-tuning beat retrieval, and do they compose?" is a question teams argue about with no data. Report accuracy, latency, and cost for each.

Depth credential

src/attention.py — multi-head self-attention implemented from scratch (~200 lines), verified numerically against HuggingFace's implementation in a unit test. Lets her honestly claim she has implemented a transformer, not only called one.

Day-by-day (~2 weeks)
Load QASPER, understand its structure, write the evaluation harness first (before any system exists) — same discipline as project 1.
BM25 retrieval baseline + retrieval recall@k. First real number.
Dense embeddings + FAISS index; compare against BM25; try hybrid.
Generation with Qwen2.5-1.5B; answer-F1; end-to-end pipeline working.
Unanswerable-question hallucination rate + failure decomposition. The core result.
Calibration — logprob confidence, ECE, does confidence predict correctness.
LoRA fine-tune on Kaggle.
Four-way comparison (base/FT × with/without retrieval).
FastAPI service + Dockerfile + unit tests + GitHub Actions.
AWS deployment (ECR + Lambda container or EC2 free tier), CloudWatch, cost/latency per query.
attention.py from scratch + numerical equivalence test.
README, figures, results CSVs committed, GitHub push.
Keywords honestly earned

LLM · RAG · retrieval-augmented generation · embeddings · vector database (FAISS) · LoRA / PEFT · fine-tuning · transformers · HuggingFace · PyTorch · prompt engineering · model evaluation · benchmarking · hallucination measurement · calibration · FastAPI · REST API · Docker · containerization · AWS (Lambda/EC2/S3/ECR/CloudWatch) · CI/CD · GitHub Actions · MLOps · inference optimization · cost analysis

Target resume bullets (measure first, then fill in)
Built and deployed a retrieval-augmented QA system over research papers (FastAPI, Docker, AWS Lambda, FAISS) evaluated on QASPER, achieving X answer-F1 while measuring a Y% fabrication rate on questions the source paper does not answer.
Decomposed system failures into retrieval vs generation, showing Z% of errors occurred despite correct evidence retrieval — isolating the generation step as the bottleneck.
LoRA-fine-tuned Qwen2.5-1.5B (~1% of parameters trainable) on free-tier GPUs and benchmarked four configurations (base/fine-tuned × with/without retrieval) on accuracy, latency, and cost per query.
Implemented multi-head self-attention from scratch, verified numerically against the HuggingFace reference.
Still outstanding (higher leverage than any project)
Email UMass professors about undergraduate research, attaching the art-detection repo.
Find a real user for a document-search tool (a lab, department, or student org) — "adopted by X" is the line peer resumes have and she doesn't.
Diagnose the 0-from-100 application round: timing (ML internships open Aug–Oct), role targeting (MLE/Applied Scientist postings filter for MS/PhD), referrals, ATS-parseable resume formatting.