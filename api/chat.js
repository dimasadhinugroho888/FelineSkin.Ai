const OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions";

const CHAT_MODEL_CANDIDATES = [
  "nvidia/nemotron-3-ultra-550b-a55b:free",
  "openai/gpt-oss-120b:free",
  "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
  "qwen/qwen3-next-80b-a3b-instruct:free",
  "mistralai/mistral-7b-instruct:free",
];

const FELINA_SYSTEM_PROMPT = `Kamu adalah Felina, asisten virtual AI yang ramah dan berpengetahuan luas khusus tentang kucing, dibuat oleh FelineSkin.AI.

Kepribadianmu:
- Hangat, bersahabat, dan antusias terhadap kucing
- Berbicara dalam Bahasa Indonesia yang santai dan mudah dipahami
- Sering menggunakan emoji kucing 🐱 atau 🐾 di respon kamu
- Memperkenalkan diri sebagai "Felina" jika ditanya siapa kamu

Batasan KETAT yang WAJIB kamu ikuti:
- Kamu HANYA boleh menjawab pertanyaan yang berkaitan dengan kucing (kesehatan kucing, perawatan kucing, penyakit kucing, nutrisi kucing, perilaku kucing, ras kucing, dll.)
- Jika pengguna bertanya tentang hal yang TIDAK berhubungan dengan kucing (politik, memasak, matematika, teknologi umum, dll.), tolak dengan sopan dan arahkan kembali ke topik kucing.
- Kamu bukan dokter hewan. Selalu sarankan konsultasi ke dokter hewan profesional untuk diagnosis dan pengobatan resmi.
- Jangan membuat diagnosis medis definitif.
- Maksimal 250 kata per respons. Gunakan format poin jika informasinya banyak.

Mulai setiap percakapan dengan ramah!`;

async function tryChatWithModel(apiKey, model, messages) {
  const response = await fetch(OPENROUTER_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      "X-Title": "FelineSkin.AI - Felina Chatbot",
    },
    body: JSON.stringify({
      model,
      messages,
      temperature: 0.7,
      max_tokens: 600,
    }),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status} from model ${model}`);
  }

  const data = await response.json();
  const text = data?.choices?.[0]?.message?.content;
  if (!text) throw new Error(`Empty response from model ${model}`);

  return text;
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

  const { messages } = req.body || {};
  if (!Array.isArray(messages) || messages.length === 0) {
    res.status(400).json({ error: "Payload tidak valid: messages harus berupa array" });
    return;
  }

  // Build conversation with system prompt
  const conversation = [
    { role: "system", content: FELINA_SYSTEM_PROMPT },
    ...messages.slice(-10), // keep last 10 messages for context
  ];

  const failures = [];
  for (const model of CHAT_MODEL_CANDIDATES) {
    try {
      const reply = await tryChatWithModel(apiKey, model, conversation);
      res.status(200).json({ reply, model });
      return;
    } catch (err) {
      failures.push(`${model}: ${err.message}`);
    }
  }

  res.status(503).json({
    reply: "Maaf, Felina sedang tidak bisa dijangkau saat ini. Silakan coba lagi nanti! 🐱",
    failures,
  });
};
