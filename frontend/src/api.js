import axios from "axios";

const API_URL = "http://localhost:5000";

export const uploadImage = async (file) => {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await axios.post(`${API_URL}/upload`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });

    // Convert base64 to Blob
    const output_2d = base64ToBlob(response.data.output_2d, 'image/jpeg');
    const output_3d = base64ToBlob(response.data.output_3d, 'image/jpeg');

    return { output_2d, output_3d };
  } catch (error) {
    console.error("Upload error:", error.response?.data || error.message);
    throw error;
  }
};

// Helper function to convert base64 to Blob
function base64ToBlob(base64Str, contentType) {
  const byteCharacters = atob(base64Str);
  const byteNumbers = new Array(byteCharacters.length);
  
  for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i);
  }
  
  const byteArray = new Uint8Array(byteNumbers);
  return new Blob([byteArray], { type: contentType });
}