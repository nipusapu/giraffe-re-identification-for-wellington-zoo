export const runtime = 'nodejs';
// (optional) ensure no caching during polling
export const dynamic = 'force-dynamic';

type RouteCtx = { params: Promise<{ id: string }> };

export async function GET(_req: Request, ctx: RouteCtx) {
  const { id } = await ctx.params; // <-- await params

  const base = (process.env.DJANGO_API_BASE ?? '').replace(/\/+$/, '');
  const url = `${base}/api/result/${encodeURIComponent(id)}/`;

  const upstream = await fetch(url, {
    headers: { 'X-API-Key': process.env.REID_API_KEY ?? '' },
    cache: 'no-store',
  });

  const data = await upstream.json().catch(() => ({}));
  return new Response(JSON.stringify(data), { status: upstream.status });
}
