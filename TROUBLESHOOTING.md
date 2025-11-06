# Troubleshooting Guide - RICo Phase 2

Last Updated: 2025-11-06

---

## MediaPipe Installation Failure (Python 3.13+)

**Context**: TASK 2A.1 - Installing Phase 2 dependencies
**Symptom**: pip install mediapipe fails with "Could not find a version that satisfies the requirement"
**Error Snippet**: ERROR: Could not find a version that satisfies the requirement mediapipe==0.10.8 (from versions: none)
**Probable Cause**: MediaPipe incompatible with Python 3.12+ (requires Python 3.8-3.11)
**Quick Fix**: Downgrade Python to 3.10 or 3.11
**Permanent Fix**: Use Python 3.10 (recommended for MediaPipe compatibility)
**Prevention**: Check Python version before Phase 2 development

---

## Dependency Installation Issues

**Context**: Installing MediaPipe, OpenCV for Phase 2
**Symptom**: pip install fails
**Error Snippet**: [actual error]
**Probable Cause**: Python version incompatibility / binary build issues
**Quick Fix**: pip install --no-binary mediapipe mediapipe
**Permanent Fix**: Use Python 3.10 (not 3.12+), document working environment
**Prevention**: Test in clean venv before production install

---

## Mouth Tracker Low Detection Rate

**Context**: test_mouth_tracker_standalone.py fails
**Symptom**: Detection rate <50%
**Error Snippet**: AssertionError: Detection rate too low: 35.0%
**Probable Cause**: Video quality / MediaPipe confidence threshold too high
**Quick Fix**: Lower min_detection_confidence to 0.3
**Permanent Fix**: Tune thresholds per clip, handle occlusion gracefully
**Prevention**: Test on multiple clips before claiming success

---

## Phase 2 Integration Breaks Phase 1

**Context**: After adding RICo to chat_server.py
**Symptom**: Phase 1 mode no longer works
**Error Snippet**: [actual error from server logs]
**Probable Cause**: Import error / syntax error / removed Phase 1 code
**Quick Fix**: git checkout src/chat_server.py (rollback)
**Permanent Fix**: NEVER remove Phase 1 code, only add Phase 2 as optional
**Prevention**: Test Phase 1 mode IMMEDIATELY after any chat_server.py change

---

## 0% Detection Rate - Systemic Face Detection Failure

**Context:** Any pipeline using OpenCV+MediaPipe fails to detect any faces, even on clean sample video.
**Symptom:** 0.0% detection, no exceptions. MediaPipe reports faces found but visibility confidence is 0.0.
**Probable Causes:**
- Video file/path mismatch or corrupt
- OpenCV not reading video frames properly (often all-black)
- MediaPipe face mesh not initialized; dependency error
- Frame/image dtype or shape mismatch
- **Test video contains animated avatars (not real faces)** - visibility confidence = 0.0
**Quick Fix:** Print and save raw frames, test with known image file, step-by-step debugging to isolate problem.
**Permanent Fix:** Standardize lab validation step on an image known to work for MediaPipe on your environment.

## Mouth Tracker Test Failure - No Face Detection

**Context**: TASK 2A.2 - Testing MouthROITracker standalone
**Symptom**: Test fails with detection rate 0.0%, all frames show "Mouth occluded (confidence: 0.00)"
**Error Snippet**: AssertionError: Detection rate too low: 0.0%
**Probable Cause**: Test videos contain animated avatars where MediaPipe face mesh visibility confidence = 0.0
**Quick Fix**: Lower mouth visibility threshold from 0.3 to 0.0, or use videos with real human faces
**Permanent Fix**: Adjust confidence thresholds for animated content, or source real face videos
**Prevention**: Test MediaPipe visibility confidence directly, not just face detection

## Fallback Path Not Working

**Context**: Phase 2 fails but Phase 1 doesn't activate
**Symptom**: Server crashes or returns error to user
**Error Snippet**: [actual exception]
**Probable Cause**: Missing try/except, video_path variable not set
**Quick Fix**: Ensure video_path = None before Phase 2, then check if None for fallback
**Permanent Fix**: Explicit if video_path is None: run Phase 1
**Prevention**: Test fallback explicitly (force Phase 2 to fail, verify Phase 1 runs)
