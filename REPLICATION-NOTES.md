# Replication Notes - RICo Phase 2

Last Updated: 2025-11-06

---

## Environment

- **Machine**: Z8 Workstation
- **OS**: Linux
- **Python**: 3.13.5 (INCOMPATIBLE with MediaPipe)
- **Working Branch**: rico-phase1-complete (baseline)

---

## Known Pitfalls to Avoid Next Run

1. ❌ **DO NOT** install MediaPipe on Python 3.12+ (incompatible)
2. ❌ **DO NOT** remove Phase 1 code when adding Phase 2
3. ❌ **DO NOT** commit changes to chat_server.py without testing Phase 1 mode first
4. ✅ **DO** test each component in isolation before integration
5. ✅ **DO** verify fallback path explicitly (force errors)
6. ✅ **DO** keep ENABLE_RICO_PHASE2 default as false
7. ✅ **DO** use Python 3.10 or 3.11 venv for Phase 2 development (not system Python 3.13)

---

## Replicable Setup Checklist

- [ ] Git branch: rico-phase1-complete (verified working)
- [ ] Python 3.10 in clean venv (NOT 3.13!)
- [ ] pip install -r requirements.txt
- [ ] Test Phase 1: python -m uvicorn src.chat_server:app
- [ ] Browser test: Send message, verify video plays
- [ ] pip install mediapipe opencv-python (Phase 2 deps)
- [ ] Test imports: python -c "import mediapipe, cv2"
- [ ] Create isolated components (mouth_tracker, etc)
- [ ] Test each component standalone
- [ ] Add feature flag to chat_server.py
- [ ] Test Phase 1 mode after flag (must still work)
- [ ] Add RICo integration with try/except
- [ ] Test fallback path (force Phase 2 error)
- [ ] Final validation: Phase 1 and Phase 2 modes

---

## Flaky Tests

None documented yet.

---

## Recurring Errors

- MediaPipe installation fails on Python 3.13+
- Solution: Use Python 3.10 or 3.11
- Mouth tracker test fails with 0.0% detection rate on existing video clips
- Solution: Use videos with clear, detectable human faces (existing clips may be animated avatars)
