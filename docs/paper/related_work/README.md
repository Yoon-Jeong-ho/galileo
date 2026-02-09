# Related Work Vault (GALILEO)

목적:
- 우리와 유사/인접한 논문을 **한 곳에서 검색/요약/추적**하기.
- 논문 본문(특히 Related Work, Positioning)에서 인용할 때, 여기에서 먼저 확인하고 최신 요약을 유지.

원칙:
- 논문마다 **하나의 파일**: `docs/paper/related_work/papers/<slug>.md`
- 요약은 “우리 논문에 직접 도움이 되는 형태”로만: 주장/설정/지표/한계/우리와의 관계.
- 가능하면 arXiv/ACL Anthology/공식 페이지 링크 + bibtex 포인터 포함.

Index:
- 전체 논문 목록/태그/우리와의 매핑은 `INDEX.md`를 사용.

추가 규칙(검색):
- 검색은 rate limit을 고려해 **적은 쿼리로 넓게** 시작하고, 필요 시 깊게 파는 식으로 진행.
