import { NextRequest, NextResponse } from "next/server";

// Allowed trusted hosts for downloading upscaled images (SSRF prevention)
function isAllowedHost(hostname: string): boolean {
  const lowerHost = hostname.toLowerCase();
  
  // Reject IP literals (IPv4 and IPv6)
  if (/^(\d{1,3}\.){3}\d{1,3}$/.test(lowerHost) || lowerHost.includes(":")) {
    return false;
  }

  // Reject local and internal hosts
  if (
    lowerHost === "localhost" ||
    lowerHost.endsWith(".local") ||
    lowerHost.endsWith(".internal")
  ) {
    return false;
  }

  // Allowed CDN / Storage providers
  if (
    lowerHost.endsWith(".r2.cloudflarestorage.com") ||
    lowerHost.endsWith(".r2.dev") ||
    lowerHost.endsWith(".runpod.net")
  ) {
    return true;
  }

  // Check custom allowed origins from env if configured
  const customOrigins = process.env.ALLOWED_DOWNLOAD_ORIGINS;
  if (customOrigins) {
    const origins = customOrigins.split(",").map((o) => o.trim().toLowerCase());
    return origins.some((orig) => lowerHost === orig || lowerHost.endsWith(`.${orig}`));
  }

  return false;
}

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const targetUrl = searchParams.get("url");
    const filename = searchParams.get("filename") || "upscaled.jpg";

    if (!targetUrl) {
      return NextResponse.json({ error: "Missing url parameter" }, { status: 400 });
    }

    let parsedUrl: URL;
    try {
      parsedUrl = new URL(targetUrl);
    } catch {
      return NextResponse.json({ error: "Invalid URL format" }, { status: 400 });
    }

    // SSRF Check: HTTPS only & strict domain whitelist
    if (parsedUrl.protocol !== "https:") {
      return NextResponse.json({ error: "Only HTTPS URLs are permitted" }, { status: 400 });
    }

    if (!isAllowedHost(parsedUrl.hostname)) {
      return NextResponse.json({ error: "Origin host not authorized for download" }, { status: 403 });
    }

    // Fetch the image from Cloudflare R2 / storage (rejecting redirects to prevent SSRF bypass)
    const response = await fetch(targetUrl, { redirect: "error" });
    if (!response.ok) {
      return NextResponse.json(
        { error: `Failed to fetch target URL: ${response.statusText}` },
        { status: response.status }
      );
    }

    // Enforce 50MB max file size to protect server memory
    const contentLength = response.headers.get("content-length");
    const MAX_SIZE_BYTES = 50 * 1024 * 1024; // 50MB
    if (contentLength && parseInt(contentLength, 10) > MAX_SIZE_BYTES) {
      return NextResponse.json({ error: "Requested image exceeds 50MB limit" }, { status: 413 });
    }

    const contentType = response.headers.get("content-type") || "image/jpeg";

    // Set headers to force download (attachment)
    const headers = new Headers();
    headers.set("Content-Disposition", `attachment; filename="${encodeURIComponent(filename)}"`);
    headers.set("Content-Type", contentType);
    headers.set("Cache-Control", "public, max-age=31536000, immutable");

    // Stream the body directly to conserve serverless memory
    return new NextResponse(response.body, {
      status: 200,
      headers,
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Internal Server Error";
    console.error("[Download Proxy Error]:", err);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
