import { NextRequest, NextResponse } from "next/server";

export const maxDuration = 60; // Allow long-running GPU operations

export async function POST(req: NextRequest) {
  try {
    const { 
      image, 
      scale = 4, 
      face_enhance = false, 
      remove_bg = false, 
      model = "x4plus",
      userApiKey,
      userEndpointId
    } = await req.json();

    if (!image) {
      return NextResponse.json({ error: "No image provided" }, { status: 400 });
    }

    // Prioritize user's own RunPod API Key, fallback to server key if configured
    const apiKey = (userApiKey && userApiKey.trim()) || process.env.RUNPOD_API_KEY;
    const endpointId = (userEndpointId && userEndpointId.trim()) || process.env.RUNPOD_ENDPOINT_ID || "3j67gpfsvuwgy3";

    if (!apiKey) {
      return NextResponse.json(
        { error: "กรุณาใส่ RunPod API Key ของคุณในเมนู Settings ก่อนเริ่มใช้งานครับ (BYOK)" }, 
        { status: 401 }
      );
    }

    // Generate unique R2 key so the RunPod worker can upload directly to Cloudflare R2
    const imageFormat = remove_bg ? "png" : "jpg";
    const randomId = Math.random().toString(36).substring(2, 12);
    const r2Key = `upscaled/batch_${Date.now()}_${randomId}.${imageFormat}`;

    // Call RunPod Serverless runsync
    const url = `https://api.runpod.ai/v2/${endpointId}/runsync`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        input: {
          image,
          scale: Number(scale),
          face_enhance: Boolean(face_enhance),
          remove_bg: Boolean(remove_bg),
          model,
          image_format: imageFormat,
          r2_key: r2Key,
        },
      }),
    });

    if (!response.ok) {
      const errText = await response.text();
      return NextResponse.json({ error: `RunPod API error: ${errText}` }, { status: response.status });
    }

    const data = await response.json();

    if (data.status === "COMPLETED" && data.output) {
      if (data.output.error) {
        return NextResponse.json({ error: data.output.error }, { status: 500 });
      }
      return NextResponse.json({
        success: true,
        r2Url: data.output.r2Url,
        width: data.output.output_size?.width || data.output.width,
        height: data.output.output_size?.height || data.output.height,
        fileSize: data.output.output_file_size || data.output.fileSize,
      });
    }

    return NextResponse.json({ error: data.status || "Failed to process image" }, { status: 500 });
  } catch (error: any) {
    return NextResponse.json({ error: error.message || "Internal server error" }, { status: 500 });
  }
}
