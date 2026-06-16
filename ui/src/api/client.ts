const BASE = '/api'

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: body !== undefined ? { 'Content-Type': 'application/json' } : {},
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error((err as { detail?: string }).detail ?? `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  get:    <T>(path: string)                  => req<T>('GET',    path),
  post:   <T>(path: string, body: unknown)   => req<T>('POST',   path, body),
  put:    <T>(path: string, body: unknown)   => req<T>('PUT',    path, body),
  delete: <T>(path: string)                  => req<T>('DELETE', path),
}
