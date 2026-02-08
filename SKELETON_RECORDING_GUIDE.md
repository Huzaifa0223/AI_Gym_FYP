# 🎥 SKELETON VIDEO RECORDING - QUICK GUIDE

## ✨ NEW FEATURE: Post-Session Skeleton Videos

After your workout, you can now create **skeleton-only videos** to analyze your body movements without showing yourself on camera!

---

## 🎬 THREE WAYS TO RECORD

### **1. AUTO-RECORD (Easiest)** ⭐ RECOMMENDED
```powershell
# Record for 60 seconds (default)
python auto_skeleton_recorder.py

# Record for 2 minutes
python auto_skeleton_recorder.py --duration 120

# Custom duration
python auto_skeleton_recorder.py --duration 30
```

**What happens:**
1. 3-second countdown
2. Automatically starts recording
3. Shows live preview (with person)
4. Saves skeleton-only video (no person, just skeleton on black background)
5. Stops after time expires

**Perfect for:**
- Quick workout recording
- Timed exercise sessions
- Automatic post-session analysis

---

### **2. MANUAL RECORDING**
```powershell
python skeleton_recorder.py
```

**Controls:**
- Press `R` to start recording
- Do your exercise
- Press `R` again to stop
- Video automatically saves

**Perfect for:**
- Variable-length sessions
- Multiple recordings in one session
- More control over recording

---

### **3. LIVE SKELETON VIEW (No Recording)**
```powershell
python skeleton_demo.py
```

**Just shows skeleton overlay - no recording**

---

## 📁 OUTPUT FILES

### **Where videos are saved:**
```
skeleton_recordings/
├── session_20260207_120530/
│   └── skeleton_only_20260207_120530.mp4  ← Your video!
├── session_20260207_121045/
│   └── skeleton_only_20260207_121045.mp4
└── ...
```

### **What the video contains:**
- ✅ Black background
- ✅ Green skeleton points (joints)
- ✅ Cyan connection lines (bones)
- ✅ Smooth 30 FPS
- ✅ Full resolution (1280x720)
- ❌ NO person visible (privacy-friendly!)

---

## 🎯 USE CASES

### **1. Check Body Joint Tracking Quality**
Record yourself doing an exercise, then watch the skeleton video to see:
- Are all joints being tracked?
- Is tracking smooth or jittery?
- Which poses work best for detection?

### **2. Analyze Movement Patterns**
Compare skeleton videos of:
- Good form vs bad form
- Before training vs after training
- Different exercise variations

### **3. Teaching/Demonstration**
Show the skeleton video to:
- Explain how pose detection works
- Demonstrate body landmark tracking
- Privacy-friendly workout sharing

### **4. Debugging**
When exercise detection isn't working:
- Record skeleton video
- Check if key joints are visible
- Identify tracking issues

---

## 🎤 FOR YOUR SUPERVISOR MEETING

### **Demo Script:**

**1. Explain the feature:**
```
"We can record skeleton-only videos for post-session analysis. 
This helps analyze body movements without exposing the person's identity."
```

**2. Run the auto-recorder:**
```powershell
python auto_skeleton_recorder.py --duration 30
```

**3. Do some bicep curls for 30 seconds**

**4. Show the output:**
```
"Now we have a video showing ONLY the skeleton movements. 
Perfect for analyzing joint tracking quality and movement patterns."
```

**5. Play the video:**
```powershell
# Open the video file in any player
# Or just drag and drop into VLC/Windows Media Player
```

---

## 📊 WHAT YOU CAN ANALYZE FROM SKELETON VIDEOS

### **Joint Visibility:**
- Are shoulders, elbows, wrists always visible?
- Which body positions lose tracking?
- Best camera angles for each exercise

### **Movement Smoothness:**
- Is skeleton jittery or smooth?
- Tracking quality at different speeds
- Effect of lighting conditions

### **Form Comparison:**
- Side-by-side skeleton videos of good vs bad form
- Visual proof of posture improvements
- Clear demonstration of technique differences

### **Exercise Signature:**
- Each exercise has unique skeleton movement pattern
- Helps improve auto-detection algorithms
- Training data visualization

---

## 💡 TIPS FOR BEST RECORDINGS

### **Camera Setup:**
- Position camera 6-8 feet away
- Camera at chest/waist height
- Side view for bicep/shoulder exercises
- Front view for chest/leg exercises

### **Lighting:**
- Bright, even lighting
- Light from front or sides (not backlit)
- Avoid shadows on body

### **Clothing:**
- Fitted clothes show body landmarks better
- Avoid baggy/oversized clothing
- Plain background helps

### **Movements:**
- Full range of motion
- Controlled speed (not too fast)
- Stay in camera frame

---

## 🔧 TECHNICAL DETAILS

### **Video Specifications:**
- **Format:** MP4 (H.264)
- **Resolution:** 1280x720
- **Frame Rate:** 30 FPS
- **Background:** Pure black (RGB: 0,0,0)
- **Skeleton Color:** Green points, Cyan lines
- **File Size:** ~2-5 MB per minute

### **Processing:**
- Real-time MediaPipe pose detection
- 33-point body model
- 3D landmark tracking
- Visibility-weighted rendering

---

## ❓ COMMON QUESTIONS

### **Q: Why skeleton-only?**
**A:** Privacy, cleaner analysis, smaller file size, focus on movements not appearance

### **Q: Can I see myself while recording?**
**A:** Yes! Live preview shows you, but saved video is skeleton-only

### **Q: How long can I record?**
**A:** As long as you want! Default is 60 seconds, but customizable

### **Q: What if skeleton isn't detected?**
**A:** Video still records (black frames). Ensure you're in camera view and well-lit

### **Q: Can I share these videos?**
**A:** Yes! They're privacy-friendly (no face/body visible)

---

## 🚀 NEXT STEPS

1. **Test it now:**
   ```powershell
   python auto_skeleton_recorder.py --duration 15
   ```

2. **Do a quick exercise** (15 seconds)

3. **Check the output:**
   ```powershell
   # Navigate to skeleton_recordings folder
   # Open the video file
   ```

4. **Use in your meeting** to demonstrate the technology!

---

## 📝 COMPARISON TABLE

| Feature | skeleton_demo.py | skeleton_recorder.py | auto_skeleton_recorder.py |
|---------|------------------|----------------------|---------------------------|
| Live View | ✅ | ✅ | ✅ |
| Records Video | ❌ | ✅ | ✅ |
| Manual Control | N/A | ✅ | ❌ |
| Auto-Stop | N/A | ❌ | ✅ |
| Best For | Quick testing | Flexible recording | Timed sessions |

---

**Perfect for your meeting! Show them the technology behind pose tracking! 🎯**
