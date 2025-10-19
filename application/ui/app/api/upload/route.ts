// app/api/upload/route.ts
export const runtime = "nodejs";

function withSlash(base: string, path: string) {
  return base.replace(/\/+$/, '') + (path.startsWith('/') ? path : `/${path}`);
}

export async function POST(req: Request) {
  const form = await req.formData();
  const file = form.get("image") as File | null;
  if (!file) return new Response(JSON.stringify({ detail: "No image provided." }), { status: 400 });

  const url = withSlash(process.env.DJANGO_API_BASE!, "/api/upload/"); // <-- NOTE THE SLASH
  const upstream = await fetch(url, {
    method: "POST",
    headers: { "X-API-Key": process.env.REID_API_KEY! },
    body: form,
  });

  const data = await upstream.json().catch(() => ({}));
  return new Response(JSON.stringify(data), { status: upstream.status });
}
