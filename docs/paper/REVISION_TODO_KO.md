# (수정/보완해야 할 것만) REVISION TODO — GALILEO (KO)

> 형식: “현재 ~~가 부족/문제이고 → ~~해서 → ~~로 수정해야 한다”
> 목적: 리뷰어 리스크(거절 사유)만 줄이는 수정 포인트를 SSOT로 관리.

1) 현재 **논문 기여/차별점이 ‘멀티턴 sycophancy/persuasion’ 기존선과 겹쳐 보일 위험**이 있고 → SYCON/TRUTH DECAY/Challenging-the-Evaluator 대비 **(i) ground-truth tasks**, (ii) survival/TOF/recovery 3축, (iii) Neutral Re-asking Control로 drift 분리 라인을 더 전면에 세워서 → 서론/초록 첫 단락을 **“우리가 하는 것/안 하는 것”**이 한 번에 보이게 수정해야 한다.

2) 현재 **Neutral Re-asking Control의 ‘no-new-evidence’ 제약이 리뷰어에게 모호**하게 읽힐 수 있고 → control 템플릿(한 문장, 대안답/새 근거 금지)의 구현 포인터를 본문에 1회 더 명시해서 → “control은 evidence-based belief revision이 아니라 generic drift 측정”이 오해 없이 고정되도록 수정해야 한다.

3) 현재 **Survival/TOF/Flip 정의가 수식/언어로 완전히 고정되지 않아**(특히 ‘round r에서 맞음’ vs ‘1..r 내내 맞음’ 혼동) → 정의를 본문(방법/메트릭)과 그림 캡션에서 **동일 문장**으로 통일해서 → reviewer가 metric을 재구성할 때 ambiguity가 없도록 수정해야 한다.

4) 현재 **TOF가 SYCON의 “turn of flip”과 용어 충돌**할 수 있고 → 본문에서 “TOF=turn-of-failure(ground-truth incorrect 최초 라운드)”를 한 번 더 강조하고 SYCON과의 관계(유사하지만 setting/score 다름)를 짧게 정리해서 → 용어 혼선을 사전에 차단하도록 수정해야 한다.

5) 현재 **정성 분석(Flip taxonomy)이 ‘계획’ 수준으로 남아** accept에 치명적일 수 있고 → seed1–4에서 persona vs control 각각 최소 N개(예: 30–50) flip 샘플을 **taxonomy bucket으로 실제 라벨링**하고 대표 사례를 표/그림으로 고정해서 → “왜 그런가”가 데이터로 보이도록 수정해야 한다.

6) 현재 **extractive QA의 EM 기반 flip이 ‘의미 변화’로 과장될 위험**이 있고 → boundary/overanswer/partial/semantic을 ‘진단용’이라고 명시하고, 메인 claim은 evaluator 기반(재현성)임을 유지하되 → 본문에서 “semantic-change가 persona에서 우세” 같은 문장은 **부록/진단에만** 한정해 과장을 줄이도록 수정해야 한다.

7) 현재 **cross-family 결과에서 모델별 max_model_len 제약(예: Nemo 32k)이 숨겨져** 비교 공정성 질문을 받을 수 있고 → Limitations에 짧게 투명 공지하고, 프로토콜/프롬프트는 동일이며 비교의 목적이 max-context가 아님을 분명히 해서 → ‘설정 차이’ 지적을 미리 방어하도록 수정해야 한다.

8) 현재 **샘플링/seed/신뢰구간(변동성) 보고가 일부 섹션에서 불균일**하고 → 모든 핵심 수치에 대해 (seed 범위, mean±std, n) 표기 규칙을 만들고 본문/캡션에 적용해서 → 통계 보고의 일관성을 확보하도록 수정해야 한다.

9) 현재 **Table W가 “무슨 표인지/왜 필요한지” 한 문장 설명이 부족**할 수 있고 → Table W를 “control vs persona를 drift-보정된 효과(Δ metric)로 요약한 1차 headline”으로 정의하고 → 결과 도입부에 Table W를 읽는 법(ΔSurvival/ΔFail@1/ΔRecovery) 2–3문장으로 추가해야 한다.

10) 현재 **Recovery가 ‘프롬프트 개입’이라 설득력이 약해 보일 위험**이 있고 → verify_then_answer 등 variant 결과를 본문에서 “intervention-dependence를 점검하는 최소 ablation”으로 위치시키고 → recovery claim을 ‘절대값’이 아니라 ‘distinct axis/상대 비교’로 제한해 주장 강도를 조정해야 한다.

11) 현재 **재현성(artifacts/validator/runner_metadata parity)의 reviewer-facing 설명이 길게 흩어져** 있고 → “한 페이지: 어떻게 재현/검증하는가”를 README/Appendix 형태로 묶고(경로/커맨드 3줄) → reviewer가 5분 안에 검증 루트를 따라갈 수 있도록 수정해야 한다.

12) 현재 **익명화(호스트 경로/모델 실행 환경) 리스크가 본문/부록에 잔존**할 수 있고 → ANONYMIZATION_NOTES 기준으로 `<REMOTE_HOST>/<REMOTE_REPO_ROOT>` 표기 규칙을 전면 적용해서 → 제출 직전 익명성 사고를 방지하도록 수정해야 한다.
