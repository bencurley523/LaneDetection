using UnityEngine;
using UnityEngine.UI;
using Unity.InferenceEngine;

public class laneDetection : MonoBehaviour
{
    [Header("Model")]
    public ModelAsset modelAsset;
    private BackendType backend = BackendType.GPUCompute; // GPUCompute | CPU
    //public string inputName = "input";    // match your ONNX input name
    private string outputName = "conv2d_29";  // match your ONNX output name
    private int inputWidth = 640;
    private int inputHeight = 384;
    [Range(0,1f)] private float threshold = 0.12f;

    [Header("Camera & Overlay")]
    private Camera sourceCamera;
    public Material overlayMaterial;      // assign a simple transparent/blend shader
    [Range(0, 1)] private float overlayAlpha = 0.6f;
    public RawImage overlayUI;

    // runtime
    private Model model;
    private Worker worker;
    private RenderTexture camRT;          // camera at model size
    private Texture2D readTex;            // CPU readback (simple path)
    private Texture2D maskTex;            // RGBA mask we draw
    private float[] inputData;            // NHWC packed floats
    private TensorShape inShape, outShape;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        sourceCamera = GameObject.Find("FrontCam").GetComponent<Camera>();

        model = ModelLoader.Load(modelAsset);
        worker = new Worker(model, backend);

        // Shapes (NHWC)
        inShape = new TensorShape(1, inputHeight, inputWidth, 3);

        // Allocate buffers
        camRT = new RenderTexture(inputWidth, inputHeight, 16, RenderTextureFormat.ARGB32);
        camRT.Create();
        readTex = new Texture2D(inputWidth, inputHeight, TextureFormat.RGBA32, false, false);
        maskTex = new Texture2D(inputWidth, inputHeight, TextureFormat.RGBA32, false, false);
        inputData = new float[inputWidth * inputHeight * 3];

        if (overlayMaterial != null)
        {
            overlayMaterial.SetTexture("_MaskTex", maskTex);
            overlayMaterial.SetFloat("_Alpha", overlayAlpha);
        }

        overlayUI.texture = maskTex;
        overlayUI.material = overlayMaterial;
    }

    void OnDestroy()
    {
        worker?.Dispose();
        if (camRT != null) camRT.Release();
        if (readTex != null) Destroy(readTex);
        if (maskTex != null) Destroy(maskTex);
    }
    
    void Update()
    {
        // 1) Render camera to low-res RT that matches model input
        var prev = sourceCamera.targetTexture;
        sourceCamera.targetTexture = camRT;
        sourceCamera.Render();
        sourceCamera.targetTexture = prev;

        // 2) Read pixels (simple & portable; optimize later with AsyncGPUReadback/compute)
        RenderTexture.active = camRT;
        readTex.ReadPixels(new Rect(0, 0, inputWidth, inputHeight), 0, 0, false);
        readTex.Apply(false);
        RenderTexture.active = null;

        // 3) Convert to NHWC tensor, normalized [-1,1] for MobileNet
        var cols = readTex.GetPixels32();
        int idx = 0;
        for (int y = 0; y < inputHeight; y++)
        {
            for (int x = 0; x < inputWidth; x++)
            {
                var c = cols[idx++];
                int baseIdx = ((y * inputWidth) + x) * 3;

                // [-1,1] normalization
                inputData[baseIdx + 0] = c.r / 127.5f - 1f;
                inputData[baseIdx + 1] = c.g / 127.5f - 1f;
                inputData[baseIdx + 2] = c.b / 127.5f - 1f;
            }
        }

        // 4) Create input tensor and run the model
        using var inputTensor = new Tensor<float>(inShape, inputData);
        worker.Schedule(inputTensor);

        // 5) Fetch output tensor
        using var outputTensor = worker.PeekOutput(outputName) as Tensor<float>;

        // 6) Convert output probabilities → pink mask
        var probs = outputTensor.DownloadToArray();
        var maskPixels = maskTex.GetPixels32();

        for (int i = 0; i < probs.Length; i++)
        {
            bool on = probs[i] > threshold;
            maskPixels[i] = on
                ? new Color32(255, 0, 255, 140)
                : new Color32(0, 0, 0, 0);
        }

        maskTex.SetPixels32(maskPixels);
        maskTex.Apply(false);
    }
    // void OnRenderImage(RenderTexture src, RenderTexture dest)
    // {
    //     // 7) Composite overlay
    //     if (overlayMaterial != null)
    //     {
    //         overlayMaterial.SetFloat("_Alpha", overlayAlpha);
    //         Graphics.Blit(src, dest, overlayMaterial);
    //     }
    //     else
    //     {
    //         Graphics.Blit(src, dest);
    //     }
    // }
}