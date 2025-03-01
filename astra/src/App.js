import React, { useState } from "react";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [imageUrl, setImageUrl] = useState("");
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [numOutputs, setNumOutputs] = useState(1);
  const [numInferenceSteps, setNumInferenceSteps] = useState(100);
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [imageGuidanceScale, setImageGuidanceScale] = useState(1.5);
  const [scheduler, setScheduler] = useState("K_EULER_ANCESTRAL");
  const [seed, setSeed] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [advancedMode, setAdvancedMode] = useState(false);

  const schedulerOptions = [
    "DDIM",
    "K_EULER",
    "DPMSolverMultistep",
    "K_EULER_ANCESTRAL",
    "PNDM",
    "KLMS",
  ];

  const handleImageUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Show preview
    const reader = new FileReader();
    reader.onload = () => {
      setImage(reader.result);
    };
    reader.readAsDataURL(file);

    // Upload to server
    setLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("http://localhost:5000/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to upload image");
      }

      const data = await response.json();
      setImageUrl(data.image_url);
      setLoading(false);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!imageUrl || !prompt) {
      setError("Please upload an image and enter a prompt");
      return;
    }

    setLoading(true);
    setError("");
    setResults([]);

    try {
      const requestData = {
        image_url: imageUrl,
        prompt: prompt,
        negative_prompt: negativePrompt || undefined,
        num_outputs: numOutputs,
        num_inference_steps: numInferenceSteps,
        guidance_scale: guidanceScale,
        image_guidance_scale: imageGuidanceScale,
        scheduler: scheduler,
        seed: seed ? parseInt(seed) : undefined,
      };

      const response = await fetch("http://localhost:5000/edit", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to process image");
      }

      const data = await response.json();
      setResults(data.result_images);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="header">
        <div className="logo">
          <span className="logo-icon"></span>
          <h1>Jewelry Image Editor</h1>
        </div>
        <p className="tagline">
          Transform your images with natural language prompts
        </p>
      </div>

      <div className="main-card">
        <form onSubmit={handleSubmit} className="edit-form">
          <div className="form-section upload-section">
            <h2>Upload Your Image</h2>
            <div className="upload-container">
              <div className="upload-box">
                {image ? (
                  <img src={image} alt="Preview" className="image-preview" />
                ) : (
                  <div className="upload-placeholder">
                    <span className="upload-icon">🖼️</span>
                    <p>Drag & drop an image or click to browse</p>
                  </div>
                )}
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="file-input"
                  id="image-upload"
                />
                <label htmlFor="image-upload" className="upload-button">
                  Select Image
                </label>
              </div>
            </div>
          </div>

          <div className="form-section prompt-section">
            <h2>Enter Your Editing Prompt</h2>
            <div className="input-group">
              <input
                type="text"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="E.g., Make this ring in Gold"
                className="text-input prompt-input"
                required
              />
            </div>
          </div>

          <div className="advanced-toggle">
            <button
              type="button"
              onClick={() => setAdvancedMode(!advancedMode)}
              className="toggle-button"
            >
              {advancedMode
                ? "▲ Hide Advanced Options"
                : "▼ Show Advanced Options"}
            </button>
          </div>

          {advancedMode && (
            <div className="advanced-options">
              <div className="input-row">
                <div className="input-group">
                  <label>Negative Prompt</label>
                  <input
                    type="text"
                    value={negativePrompt}
                    onChange={(e) => setNegativePrompt(e.target.value)}
                    placeholder="What you don't want to see"
                    className="text-input"
                  />
                </div>
              </div>

              <div className="input-row two-columns">
                <div className="input-group">
                  <label>Number of Outputs</label>
                  <select
                    value={numOutputs}
                    onChange={(e) => setNumOutputs(parseInt(e.target.value))}
                    className="select-input"
                  >
                    <option value={1}>1</option>
                    <option value={4}>4</option>
                  </select>
                </div>

                <div className="input-group">
                  <label>Scheduler</label>
                  <select
                    value={scheduler}
                    onChange={(e) => setScheduler(e.target.value)}
                    className="select-input"
                  >
                    {schedulerOptions.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="input-group">
                <label>Inference Steps: {numInferenceSteps}</label>
                <div className="slider-container">
                  <input
                    type="range"
                    min="1"
                    max="500"
                    value={numInferenceSteps}
                    onChange={(e) =>
                      setNumInferenceSteps(parseInt(e.target.value))
                    }
                    className="slider-input"
                  />
                  <div className="slider-labels">
                    <span>1</span>
                    <span>500</span>
                  </div>
                </div>
              </div>

              <div className="input-group">
                <label>Guidance Scale: {guidanceScale}</label>
                <div className="slider-container">
                  <input
                    type="range"
                    min="1"
                    max="20"
                    step="0.1"
                    value={guidanceScale}
                    onChange={(e) =>
                      setGuidanceScale(parseFloat(e.target.value))
                    }
                    className="slider-input"
                  />
                  <div className="slider-labels">
                    <span>1</span>
                    <span>20</span>
                  </div>
                </div>
              </div>

              <div className="input-group">
                <label>Image Guidance Scale: {imageGuidanceScale}</label>
                <div className="slider-container">
                  <input
                    type="range"
                    min="1"
                    max="5"
                    step="0.1"
                    value={imageGuidanceScale}
                    onChange={(e) =>
                      setImageGuidanceScale(parseFloat(e.target.value))
                    }
                    className="slider-input"
                  />
                  <div className="slider-labels">
                    <span>1</span>
                    <span>5</span>
                  </div>
                </div>
              </div>

              <div className="input-group">
                <label>Seed (Leave blank to randomize)</label>
                <input
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(e.target.value)}
                  placeholder="Random seed"
                  className="text-input"
                />
              </div>
            </div>
          )}

          <div className="submit-section">
            <button
              type="submit"
              disabled={loading || !imageUrl}
              className={`submit-button ${
                loading || !imageUrl ? "disabled" : ""
              }`}
            >
              {loading ? (
                <>
                  <span className="spinner"></span>
                  Processing...
                </>
              ) : (
                "Transform Image"
              )}
            </button>
          </div>

          {error && (
            <div className="error-message">
              <span className="error-icon">⚠️</span>
              {error}
            </div>
          )}
        </form>

        {results.length > 0 && (
          <div className="results-container">
            <h2 className="results-title">Transformation Results</h2>

            <div className="comparison-container">
              <div className="image-card original-image">
                <div className="card-header">
                  <h3>Original Image</h3>
                </div>
                <div className="card-body">
                  <img src={image} alt="Original" />
                </div>
              </div>

              {results.map((imageUrl, index) => (
                <div key={index} className="image-card result-image">
                  <div className="card-header">
                    <h3>
                      Transformed {results.length > 1 ? `#${index + 1}` : ""}
                    </h3>
                  </div>
                  <div className="card-body">
                    <img src={imageUrl} alt={`Result ${index + 1}`} />
                  </div>
                  <div className="card-footer">
                    <a
                      href={imageUrl}
                      download={`transformed-image-${index + 1}.png`}
                      className="download-button"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Download Image
                    </a>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <footer className="app-footer"></footer>
    </div>
  );
}

export default App;
