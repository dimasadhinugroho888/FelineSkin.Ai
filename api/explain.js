const OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions";

const MODEL_CANDIDATES = [
  "deepseek/deepseek-v4-flash:free",
  "google/gemma-3-27b-it:free",
  "mistralai/mistral-7b-instruct:free",
];

function buildPrompt({ diseaseName, confidence }) {
  return `Konteks: hasil klasifikasi penyakit kulit kucing adalah "${diseaseName}" dengan confidence ${confidence.toFixed(2)}%.

Berikan jawaban ringkas berbahasa Indonesia dengan format:
1) Penjelasan singkat
2) Penyebab umum
3) Gejala penting
4) Penanganan awal di rumah
5) Kapan harus ke dokter hewan

Catatan:
- Jangan klaim diagnosis pasti.
- Gunakan bahasa sederhana untuk pemilik kucing awam.
- Maksimal 220 kata.`;
}

async function requestOpenRouter(apiKey, prompt) {
  for (const model of MODEL_CANDIDATES) {
    try {
      const response = await fetch(OPENROUTER_URL, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: prompt }],
          temperature: 0.3,
        }),
      });

      if (!response.ok) {
        continue;
      }

      const data = await response.json();
      const text = data?.choices?.[0]?.message?.content;
      if (text) {
        return { ok: true, text };
      }
    } catch (_err) {
      // try next model
    }
  }

  return { ok: false, text: "AI explanation sedang tidak tersedia. Silakan coba lagi beberapa saat." };
}

module.exports = async function handler(req, res) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    res.status(500).json({ error: "OPENROUTER_API_KEY belum diset di environment" });
    return;
  }

  const { diseaseName, confidence } = req.body || {};
  if (!diseaseName || Number.isNaN(Number(confidence))) {
    res.status(400).json({ error: "Payload tidak valid" });
    return;
  }

  const prompt = buildPrompt({ diseaseName, confidence: Number(confidence) });
  const result = await requestOpenRouter(apiKey, prompt);

  if (!result.ok) {
    res.status(503).json({ explanation: result.text });
    return;
  }

  res.status(200).json({ explanation: result.text });
};
