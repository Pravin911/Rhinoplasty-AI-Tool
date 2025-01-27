import React from "react";

function Visualizer({ output2D, output3D }) {
  return (
    <div>
      <h2>2D Output</h2>
      <img src={output2D} alt="2D Nose Reshaping" />
      <h2>3D Output</h2>
      <img src={output3D} alt="3D Face Reconstruction" />
    </div>
  );
}

export default Visualizer;