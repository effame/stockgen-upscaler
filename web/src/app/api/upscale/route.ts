import { NextRequest, NextResponse } from "next/server";

export const maxDuration = 60; // Allow long-running GPU operations

interface RunPodJobOutput {
  error?: string;
  r2Url?: string;
  output_size?: { width: number; height: number };
  width?: number;
  height?: number;
  output_file_size?: number;
  fileSize?: number;
}

interface RunPodResponse {
  id?: string;
  status: "COMPLETED" | "IN_PROGRESS" | "IN_QUEUE" | "FAILED" | "CANCELLED" | string;
  output?: RunPodJobOutput;
  error?: string;
}

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

    // Require user's own RunPod API Key (BYOK) for security & quota management
    const apiKey = typeof userApiKey === "string" ? userApiKey.trim() : "";
    if (!apiKey) {
      return NextResponse.json(
        { error: "กรุณาใส่ RunPod API Key ของคุณในเมนู Settings ก่อนเริ่มใช้งานครับ (BYOK required)" }, 
        { status: 401 }
      );
    }

    const endpointId = (typeof userEndpointId === "string" && userEndpointId.trim()) 
      ? userEndpointId.trim() 
      : (process.env.RUNPOD_ENDPOINT_ID || "3j67gpfsvuwgy3");

    // Generate unique R2 key so the RunPod worker can upload directly to Cloudflare R2
    const imageFormat = remove_bg ? "png" : "jpg";
    const randomId = Math.random().toString(36).substring(2, 12);
    const r2Key = `upscaled/batch_${Date.now()}_${randomId}.${imageFormat}`;

    // Call RunPod Serverless runsync
    const runsyncUrl = `https://api.runpod.ai/v2/${endpointId}/runsync`;
    const response = await fetch(runsyncUrl, {
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

    let data: RunPodResponse = await response.json();

    // If job is in queue or in progress, poll until completion or timeout (up to 45s)
    if ((data.status === "IN_PROGRESS" || data.status === "IN_QUEUE") && data.id) {
      const jobId = data.id;
      const statusUrl = `https://api.runpod.ai/v2/${endpointId}/status/${jobId}`;
      const startTime = Date.now();
      const timeoutMs = 45000; // 45 seconds polling limit

      while (Date.now() - startTime < timeoutMs) {
        await new Promise((resolve) => setTimeout(resolve, 2000));
        
        try {
          const pollRes = await fetch(statusUrl, {
            headers: { Authorization: `Bearer ${apiKey}` },
          });
          if (pollRes.ok) {
            data = await pollRes.json();
            if (data.status === "COMPLETED" || data.status === "FAILED" || data.status === "CANCELLED") {
              break;
            }
          }
        } catch {
          // Continue polling on transient network error
        }
      }
    }

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

    if (data.status === "IN_PROGRESS" || data.status === "IN_QUEUE") {
      return NextResponse.json({ 
        error: `Job still processing in RunPod queue (Job ID: ${data.id})`, 
        jobId: data.id, 
        status: data.status 
      }, { status: 202 });
    }

    return NextResponse.json({ 
      error: data.error || data.status || "Failed to process image",
      jobId: data.id 
    }, { status: 500 });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Internal server error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
