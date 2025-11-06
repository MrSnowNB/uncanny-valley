## RICo Integration Test Produces Video But No Mouth Sync

**Context**: MVP-1.2 visual verification - output video has no mouth synchronization
**Symptom**: Video plays but mouth is static/unchanged from base video
**Observed**: Video duration correct, has audio, but mouth movements don't match speech
**Probable Cause**:
- ROI compositor not actually replacing mouth regions in frames
- Per-frame processing loop not applying viseme changes
- Pipeline using base video frames directly without compositing
- Compositor function exists but returns original frame unchanged
**Quick Fix**: Add per-frame visual debug output to verify compositing
**Permanent Fix**: Add unit test for compositor with known input/output pairs
**Prevention**: Require visual diff between base frame and composited frame in tests
