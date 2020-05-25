using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Unity.Barracuda;
using System.IO;

public class Inference : MonoBehaviour
{
    [System.Serializable]
    public class InputData
    {
        public int resolutionY;
        public int resolutionX;
        public float[] data;
    }

    private Model m_RuntimeModel;
    private IWorker m_Worker;
    private Tensor m_Input;


    public NNModel inputModel;
    public Material outputMaterial;
    public Text outputResult;


    Texture2D CreateInputTexture(InputData inputData)
    {
        var tex = new Texture2D(inputData.resolutionY, inputData.resolutionX, TextureFormat.R8, mipChain: false);
        for (int y = 0; y < tex.height; ++y)
            for (int x = 0; x < tex.width; ++x)
            {
                float v = inputData.data[x + 28 * y] / 255.0f;
                tex.SetPixel(x, tex.height - 1 - y, new Color(v,v,v));
            }
        tex.Apply();
        return tex;
    }

    void Start()
    {
        Application.targetFrameRate = 60;

        string json = File.ReadAllText(Application.dataPath + "/Images/input0.json");
        InputData inputData = JsonUtility.FromJson<InputData>(json);

        m_RuntimeModel = ModelLoader.Load(inputModel, false);
        m_Worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Compute, m_RuntimeModel, false);

        m_Input = new Tensor(1, inputData.resolutionY, inputData.resolutionX, 1, inputData.data);

        Texture2D tex = CreateInputTexture(inputData);
        outputMaterial.mainTexture = tex;
    }

    void Update()
    {
        m_Worker.Execute(m_Input);
        Tensor result = m_Worker.PeekOutput("Plus422_Output_0");
      
        int[] indicies = result.ArgMax();
        outputResult.text = "Result: " + indicies[0].ToString();
    }
}
