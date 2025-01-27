// import React, { useState } from "react";
// import { uploadImage } from "./api";
// import "./App.css";

// function App() {
//   const [file, setFile] = useState(null);
//   const [output2D, setOutput2D] = useState(null);
//   const [output3D, setOutput3D] = useState(null);
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState(null);

//   const handleFileChange = (e) => {
//     setFile(e.target.files[0]);
//   };

//   const handleSubmit = async () => {
//     if (!file) return;

//     setLoading(true);
//     setError(null);

//     try {
//       const response = await uploadImage(file);
//       const imageUrl2D = URL.createObjectURL(response.output_2d);
//       const imageUrl3D = URL.createObjectURL(response.output_3d);
//       setOutput2D(imageUrl2D);
//       setOutput3D(imageUrl3D);
//     } catch (error) {
//       console.error("Error uploading file:", error);
//       setError("Failed to process the image. Please try again.");
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <div className="container">
//       <h1>Rhinoplasty Outcome Visualizer</h1>
//       <input type="file" accept="image/*" onChange={handleFileChange} />
//       <button onClick={handleSubmit} disabled={loading}>
//         {loading ? "Generating..." : "Generate"}
//       </button>

//       {error && <p className="error">{error}</p>}

//       {output2D && output3D && (
//         <div className="output">
//           <h2>2D Output</h2>
//           <img src={output2D} alt="2D Nose Reshaping" />
//           <h2>3D Output</h2>
//           <img src={output3D} alt="3D Face Reconstruction" />
//         </div>
//       )}
//     </div>
//   );
// }

// export default App;


// import React, { useState, useEffect } from 'react';
// import axios from 'axios';
// import './App.css';

// const API_URL = 'http://localhost:5000/api';

// function App() {
//   const [file, setFile] = useState(null);
//   const [preview, setPreview] = useState(null);
//   const [output2D, setOutput2D] = useState(null);
//   const [output3D, setOutput3D] = useState(null);
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState(null);
//   const [noseStyles, setNoseStyles] = useState([]);
//   const [selectedStyle, setSelectedStyle] = useState('natural');
//   const [faceOrientation, setFaceOrientation] = useState(null);

//   // Fetch available nose styles on component mount
//   useEffect(() => {
//     axios.get(`${API_URL}/nose-styles`)
//       .then(response => setNoseStyles(response.data.styles))
//       .catch(error => console.error('Error fetching nose styles:', error));
//   }, []);

//   const handleFileChange = async (e) => {
//     const selectedFile = e.target.files[0];
//     if (!selectedFile) return;

//     setFile(selectedFile);
//     setPreview(URL.createObjectURL(selectedFile));

//     // Analyze face orientation
//     const formData = new FormData();
//     formData.append('file', selectedFile);

//     try {
//       const response = await axios.post(`${API_URL}/analyze-face`, formData);
//       setFaceOrientation(response.data.orientation);
//     } catch (error) {
//       console.error('Error analyzing face:', error);
//       setError('Failed to analyze face orientation');
//     }
//   };

//   const handleSubmit = async () => {
//     if (!file) return;

//     setLoading(true);
//     setError(null);

//     const formData = new FormData();
//     formData.append('file', file);
//     formData.append('style', selectedStyle);

//     try {
//       const response = await axios.post(`${API_URL}/generate`, formData);
//       setOutput2D(`data:image/jpeg;base64,${response.data.output_2d}`);
//       setOutput3D(`data:image/jpeg;base64,${response.data.output_3d}`);
//     } catch (error) {
//       console.error('Error generating outputs:', error);
//       setError('Failed to generate visualization');
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <div className="container">
//       <h1>Rhinoplasty Outcome Visualizer</h1>
      
//       <div className="upload-section">
//         <input 
//           type="file" 
//           accept="image/*" 
//           onChange={handleFileChange} 
//           className="file-input"
//         />
//         {preview && (
//           <img src={preview} alt="Preview" className="preview-image" />
//         )}
//       </div>

//       {faceOrientation && (
//         <div className="orientation-info">
//           <p>Detected face orientation: {faceOrientation}</p>
//         </div>
//       )}

//       <div className="style-selector">
//         <select 
//           value={selectedStyle} 
//           onChange={(e) => setSelectedStyle(e.target.value)}
//           className="style-dropdown"
//         >
//           {noseStyles.map(style => (
//             <option key={style} value={style}>
//               {style.charAt(0).toUpperCase() + style.slice(1)}
//             </option>
//           ))}
//         </select>
//       </div>

//       <button 
//         onClick={handleSubmit} 
//         disabled={loading || !file} 
//         className="generate-button"
//       >
//         {loading ? "Generating..." : "Generate Visualization"}
//       </button>

//       {error && <p className="error">{error}</p>}

//       {output2D && output3D && (
//         <div className="results">
//           <div className="result-container">
//             <h2>2D Visualization</h2>
//             <img src={output2D} alt="2D Nose Reshaping" className="result-image" />
//           </div>
//           <div className="result-container">
//             <h2>3D Visualization</h2>
//             <img src={output3D} alt="3D Face Reconstruction" className="result-image" />
//           </div>
//         </div>
//       )}
//     </div>
//   );
// }

// export default App;

// 2nd version

// import React, { useState, useEffect } from 'react';
// import { Upload, Camera, AlertCircle } from 'lucide-react';
// import './App.css';
// import axios from 'axios';

// const API_URL = 'http://localhost:5000/api';

// const RhinoplastyVisualizer = () => {
//   const [file, setFile] = useState(null);
//   const [preview, setPreview] = useState(null);
//   const [output2D, setOutput2D] = useState(null);
//   const [output3D, setOutput3D] = useState(null);
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState(null);
//   const [noseStyles, setNoseStyles] = useState([]);
//   const [selectedStyle, setSelectedStyle] = useState('natural');
//   const [faceOrientation, setFaceOrientation] = useState(null);

//   useEffect(() => {
//     axios.get(`${API_URL}/nose-styles`)
//       .then(response => setNoseStyles(response.data.styles))
//       .catch(error => {
//         console.error('Error fetching nose styles:', error);
//         setError('Failed to load nose styles');
//       });
//   }, []);

//   const handleFileChange = async (e) => {
//     const selectedFile = e.target.files[0];
//     if (!selectedFile) return;

//     if (!selectedFile.type.startsWith('image/')) {
//       setError('Please upload an image file');
//       return;
//     }

//     setFile(selectedFile);
//     setPreview(URL.createObjectURL(selectedFile));
//     setError(null);

//     const formData = new FormData();
//     formData.append('file', selectedFile);

//     try {
//       const response = await axios.post(`${API_URL}/analyze-face`, formData);
//       setFaceOrientation(response.data.orientation);
//     } catch (error) {
//       console.error('Error analyzing face:', error);
//       setError('Failed to analyze face orientation');
//     }
//   };

//   const handleSubmit = async () => {
//     if (!file) {
//       setError('Please upload an image first');
//       return;
//     }

//     setLoading(true);
//     setError(null);

//     const formData = new FormData();
//     formData.append('file', file);
//     formData.append('style', selectedStyle);

//     try {
//       const response = await axios.post(`${API_URL}/generate`, formData);
//       setOutput2D(`data:image/jpeg;base64,${response.data.output_2d}`);
//       setOutput3D(`data:image/jpeg;base64,${response.data.output_3d}`);
//     } catch (error) {
//       console.error('Error generating outputs:', error);
//       setError('Failed to generate visualization');
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <div className="app-container">
//       <h1 className="app-title">Rhinoplasty Outcome Visualizer</h1>

//       <div className="main-content">
//         {/* Input Section */}
//         <div className="card">
//           <h2 className="card-title">Upload Photo</h2>
//           <div className="card-content">
//             <div className="upload-area">
//               <label className="file-input-label">
//                 <input
//                   type="file"
//                   className="file-input"
//                   onChange={handleFileChange}
//                   accept="image/*"
//                 />
//                 {preview ? (
//                   <img
//                     src={preview}
//                     alt="Preview"
//                     className="preview-image"
//                   />
//                 ) : (
//                   <div className="upload-placeholder">
//                     <Upload size={48} className="upload-icon" />
//                     <p>Click to upload or drag and drop</p>
//                   </div>
//                 )}
//               </label>
//             </div>

//             {faceOrientation && (
//               <div className="info-box">
//                 <Camera size={20} />
//                 <span>Face orientation detected: {faceOrientation}</span>
//               </div>
//             )}

//             <select
//               value={selectedStyle}
//               onChange={(e) => setSelectedStyle(e.target.value)}
//               className="style-select"
//             >
//               {noseStyles.map(style => (
//                 <option key={style} value={style}>
//                   {style.charAt(0).toUpperCase() + style.slice(1)}
//                 </option>
//               ))}
//             </select>

//             <button
//               onClick={handleSubmit}
//               disabled={loading || !file}
//               className="submit-button"
//             >
//               {loading ? (
//                 <span className="loading-text">
//                   <div className="spinner"></div>
//                   Generating...
//                 </span>
//               ) : (
//                 "Generate Visualization"
//               )}
//             </button>

//             {error && (
//               <div className="error-message">
//                 <AlertCircle size={20} />
//                 <span>{error}</span>
//               </div>
//             )}
//           </div>
//         </div>

//         {/* Output Section */}
//         <div className="card">
//           <h2 className="card-title">Results</h2>
//           <div className="card-content">
//             {output2D && output3D ? (
//               <div className="results-container">
//                 <div className="result-item">
//                   <h3>2D Visualization</h3>
//                   <img src={output2D} alt="2D Visualization" />
//                 </div>
//                 <div className="result-item">
//                   <h3>3D Visualization</h3>
//                   <img src={output3D} alt="3D Visualization" />
//                 </div>
//               </div>
//             ) : (
//               <div className="placeholder-message">
//                 Generated visualizations will appear here
//               </div>
//             )}
//           </div>
//         </div>
//       </div>
//     </div>
//   );
// };

// 3rd Version
// export default RhinoplastyVisualizer;

import React, { useState, useEffect } from 'react';
import { Upload, Camera, AlertCircle, Info } from 'lucide-react';
import './App.css';
import axios from 'axios';

const API_URL = 'http://localhost:5000/api';

const RhinoplastyVisualizer = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [output2D, setOutput2D] = useState(null);
  const [output3D, setOutput3D] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [noseStyles, setNoseStyles] = useState([]);
  const [selectedStyle, setSelectedStyle] = useState('natural');
  const [faceOrientation, setFaceOrientation] = useState(null);
  const [styleParameters, setStyleParameters] = useState(null);
  const [landmarks, setLandmarks] = useState(null);

  useEffect(() => {
    // Fetch available nose styles
    axios.get(`${API_URL}/nose-styles`)
      .then(response => {
        setNoseStyles(response.data.styles);
        // You can also use response.data.descriptions if needed
      })
      .catch(error => {
        console.error('Error fetching nose styles:', error);
        setError('Failed to load nose styles. Please refresh the page.');
      });
  }, []);

  const handleFileChange = async (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;
  
    if (!selectedFile.type.startsWith('image/')) {
      setError('Please upload an image file');
      return;
    }
  
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setError(null);
  
    const formData = new FormData();
    formData.append('file', selectedFile);
  
    try {
      setLoading(true);
      const response = await axios.post(`${API_URL}/analyze-face`, formData);
      setFaceOrientation(response.data.orientation);
      setLandmarks(response.data.landmarks);
    } catch (error) {
      console.error('Error analyzing face:', error);
      setError(error.response?.data?.error || 'Failed to analyze face');
    } finally {
      setLoading(false);
    }
  };
  
  const handleSubmit = async () => {
    if (!file) {
      setError('Please upload an image first');
      return;
    }
  
    setLoading(true);
    setError(null);
  
    const formData = new FormData();
    formData.append('file', file);
    formData.append('style', selectedStyle);
  
    try {
      const response = await axios.post(`${API_URL}/generate`, formData);
      setOutput2D(`data:image/jpeg;base64,${response.data.output_2d}`);
      setOutput3D(`data:image/jpeg;base64,${response.data.output_3d}`);
      setStyleParameters(response.data.analysis.style_parameters);
    } catch (error) {
      console.error('Error generating outputs:', error);
      setError(error.response?.data?.error || 'Failed to generate visualization');
    } finally {
      setLoading(false);
    }
  };

  const renderStyleInfo = () => {
    if (!styleParameters) return null;

    return (
      <div className="style-info">
        <h3>Style Parameters</h3>
        <div className="parameters-grid">
          {Object.entries(styleParameters).map(([key, value]) => (
            <div key={key} className="parameter-item">
              <span className="parameter-label">{key.replace('_', ' ')}:</span>
              <span className="parameter-value">{value.toFixed(2)}</span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="app-container">
      <h1 className="app-title">Rhinoplasty Outcome Visualizer</h1>

      <div className="main-content">
        {/* Input Section */}
        <div className="card">
          <h2 className="card-title">Upload Photo</h2>
          <div className="card-content">
            <div className="upload-area">
              <label className="file-input-label">
                <input
                  type="file"
                  className="file-input"
                  onChange={handleFileChange}
                  accept="image/*"
                />
                {preview ? (
                  <img
                    src={preview}
                    alt="Preview"
                    className="preview-image"
                  />
                ) : (
                  <div className="upload-placeholder">
                    <Upload size={48} className="upload-icon" />
                    <p>Click to upload or drag and drop</p>
                    <p className="upload-hint">
                      Supports both front and side profile images
                    </p>
                  </div>
                )}
              </label>
            </div>

            {faceOrientation && (
              <div className="info-box">
                <Camera size={20} />
                <span>
                  Face orientation: {faceOrientation}
                  {faceOrientation === 'side' && (
                    <span className="info-tip">
                      <Info size={16} />
                      Side profile images will receive specialized processing
                    </span>
                  )}
                </span>
              </div>
            )}

            <div className="style-selection">
              <label htmlFor="style-select">Choose Nose Style:</label>
              <select
                id="style-select"
                value={selectedStyle}
                onChange={(e) => setSelectedStyle(e.target.value)}
                className="style-select"
              >
                {noseStyles.map(style => (
                  <option key={style} value={style}>
                    {style.charAt(0).toUpperCase() + style.slice(1)}
                  </option>
                ))}
              </select>
            </div>

            <button
              onClick={handleSubmit}
              disabled={loading || !file}
              className="submit-button"
            >
              {loading ? (
                <span className="loading-text">
                  <div className="spinner"></div>
                  Generating...
                </span>
              ) : (
                "Generate Visualization"
              )}
            </button>

            {error && (
              <div className="error-message">
                <AlertCircle size={20} />
                <span>{error}</span>
              </div>
            )}
          </div>
        </div>

        {/* Output Section */}
        <div className="card">
          <h2 className="card-title">Results</h2>
          <div className="card-content">
            {output2D && output3D ? (
              <div className="results-container">
                <div className="result-item">
                  <h3>2D Visualization</h3>
                  <img src={output2D} alt="2D Visualization" />
                  {renderStyleInfo()}
                </div>
                <div className="result-item">
                  <h3>3D Visualization</h3>
                  <img src={output3D} alt="3D Visualization" />
                </div>
              </div>
            ) : (
              <div className="placeholder-message">
                Generated visualizations will appear here
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default RhinoplastyVisualizer;