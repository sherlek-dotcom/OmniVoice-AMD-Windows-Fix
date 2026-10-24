"""OmniVoice Speech Benchmark Corpus - 8 Distinct Styles.

Covers:
1. Ultra-Short Hook (3s)
2. Technical Explainer (7s)
3. Emotion: Excited / Punchy (10s)
4. Emotion: Serious / Problem Statement (12s)
5. Phonetic Thai Normalization Stress Test (15s)
6. Emotion: Confident / Analytical (18s)
7. Long-Form Tech Narrative (25s)
8. Call-to-Action / Summary (6s)
"""

BENCHMARK_STYLES = [
    {
        "id": "style_1_hook",
        "name": "Ultra-Short Hook (3s)",
        "style_type": "hook",
        "mood": "urgent",
        "voice_text": "หยุดก่อน! ถ้าคุณกำลังคิดจะซื้อการ์ดจอมาทำ เอ ไอ ฟังคลิปนี้ให้จบ",
        "display_text": "หยุดก่อน! ถ้าคุณกำลังคิดจะซื้อการ์ดจอมาทำ AI ฟังคลิปนี้ให้จบ",
        "expected_duration_range": (2.0, 4.0),
    },
    {
        "id": "style_2_tech",
        "name": "Technical Explainer (7s)",
        "style_type": "tech_spec",
        "mood": "technical",
        "voice_text": "สถาปัตยกรรม อาร์ดีเอ็นเอ สี่ ของค่ายแดง มีหน่วยประมวลผล เทนเซอร์ คอร์ รุ่นใหม่ล่าสุด",
        "display_text": "สถาปัตยกรรม RDNA 4 ของค่ายแดง มีหน่วยประมวลผล Tensor Core รุ่นใหม่ล่าสุด",
        "expected_duration_range": (5.0, 8.5),
    },
    {
        "id": "style_3_excited",
        "name": "Emotion: Excited / Punchy (10s)",
        "style_type": "emotional_excited",
        "mood": "excited",
        "voice_text": "เฮ้ยทุกคน วันนี้ผมค้นพบวิธีปลดล็อกความเร็วในการรันโมเดลเสียงแล้ว เร็วขึ้นแบบคนละโลกเลยครับ!",
        "display_text": "เฮ้ยทุกคน วันนี้ผมค้นพบวิธีปลดล็อกความเร็วในการรันโมเดลเสียงแล้ว เร็วขึ้นแบบคนละโลกเลยครับ!",
        "expected_duration_range": (8.0, 11.5),
    },
    {
        "id": "style_4_serious",
        "name": "Emotion: Serious / Problem (12s)",
        "style_type": "emotional_serious",
        "mood": "serious",
        "voice_text": "แต่ก่อนหน้านี้มันมีคอขวดซ่อนอยู่ บั๊กของไดรเวอร์ทำเอาผมกุมขมับ รันคลิปแค่สามสิบวินาทีแต่ล่อไปเกือบสิบนาที",
        "display_text": "แต่ก่อนหน้านี้มันมีคอขวดซ่อนอยู่ บั๊กของไดรเวอร์ทำเอาผมกุมขมับ รันคลิปแค่ 30 วินาทีแต่ล่อไปเกือบ 10 นาที",
        "expected_duration_range": (10.0, 13.5),
    },
    {
        "id": "style_5_phonetic",
        "name": "Phonetic Stress Test (15s)",
        "style_type": "phonetic_stress",
        "mood": "energetic",
        "voice_text": "ถ้าคุณเป็นสายเวท ซ้อมหนักหนัก ดันเซ็ตท้ายท้ายไม่ไหว การเติมครีเอทีนสูตรไมโครไนซ์จะช่วยเสริมแรงยกได้เต็มที่",
        "display_text": "ถ้าคุณเป็นสายเวท ซ้อมหนักๆ ดันเซตท้ายๆ ไม่ค่อยไหว การเติมครีเอทีนสูตรไมโครไนซ์จะช่วยเสริมแรงยกได้เต็มที่",
        "expected_duration_range": (13.0, 16.5),
    },
    {
        "id": "style_6_confident",
        "name": "Emotion: Confident / Analytical (18s)",
        "style_type": "emotional_confident",
        "mood": "confident",
        "voice_text": "ทางแก้คือเราแยกหน้าที่กันอย่างชัดเจน ให้การ์ดจอคำนวณส่วนที่มันถนัด แล้วให้ซีพียูช่วยจัดการส่วนถอดรหัสคลื่นเสียง",
        "display_text": "ทางแก้คือเราแยกหน้าที่กันอย่างชัดเจน ให้การ์ดจอคำนวณส่วนที่มันถนัด แล้วให้ CPU ช่วยจัดการส่วนถอดรหัสคลื่นเสียง",
        "expected_duration_range": (15.0, 20.0),
    },
    {
        "id": "style_7_long_narrative",
        "name": "Long-Form Tech Narrative (25s)",
        "style_type": "long_form",
        "mood": "narrative",
        "voice_text": "นี่คือการพิสูจน์ให้เห็นว่า แม้จะอยู่บนวินโดวส์และใช้การ์ดจอค่ายแดง เราก็สามารถรีดประสิทธิภาพของโมเดลโอเพนซอร์สออกมาได้อย่างเต็มเม็ดเต็มหน่วย โดยไม่ต้องรอการอัปเดตไดรเวอร์อย่างเป็นทางการครับ",
        "display_text": "นี่คือการพิสูจน์ให้เห็นว่า แม้จะอยู่บน Windows และใช้การ์ดจอค่ายแดง เราก็สามารถรีดประสิทธิภาพของโมเดลโอเพนซอร์สออกมาได้อย่างเต็มเม็ดเต็มหน่วย โดยไม่ต้องรอการอัปเดตไดรเวอร์อย่างเป็นทางการครับ",
        "expected_duration_range": (20.0, 28.0),
    },
    {
        "id": "style_8_cta",
        "name": "Call-to-Action / Summary (6s)",
        "style_type": "cta",
        "mood": "triumphant",
        "voice_text": "กดเซฟคลิปนี้ไว้เลยครับ แล้วตามไปดูโค้ดตัวเต็มบน กิ๊ตฮับ ได้ทันที!",
        "display_text": "กดเซฟคลิปนี้ไว้เลยครับ แล้วตามไปดูโค้ดตัวเต็มบน GitHub ได้ทันที!",
        "expected_duration_range": (4.5, 7.5),
    },
]
