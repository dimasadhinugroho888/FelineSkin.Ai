const OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions";

const MODEL_CANDIDATES = [
  "openrouter/free",
  "google/gemma-4-31b-it:free",
  "google/gemma-4-26b-a4b-it:free",
  "moonshotai/kimi-k2.6:free",
  "qwen/qwen3-next-80b-a3b-instruct:free",
  "openai/gpt-oss-20b:free",
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
  const failures = [];

  for (const model of MODEL_CANDIDATES) {
    try {
      const response = await fetch(OPENROUTER_URL, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
          "X-Title": "FelineSkin.AI",
        },
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: prompt }],
          temperature: 0.3,
          max_tokens: 500,
        }),
      });

      if (!response.ok) {
        failures.push(`${model}: HTTP ${response.status}`);
        continue;
      }

      const data = await response.json();
      const text = data?.choices?.[0]?.message?.content;
      if (text) {
        return { ok: true, text };
      }

      failures.push(`${model}: empty response`);
    } catch (_err) {
      failures.push(`${model}: request failed`);
    }
  }

  return {
    ok: false,
    text: "AI explanation sedang tidak tersedia. Silakan coba lagi beberapa saat.",
    failures,
  };
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
    res.status(503).json({ explanation: result.text, details: result.failures });
    return;
  }

  res.status(200).json({ explanation: result.text });
};
