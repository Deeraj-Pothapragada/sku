import { useState, useRef } from "react";

const API_URL = "https://fluffy-ducks-fetch.loca.lt/"; // <-- paste your tunnel URL here

export default function App() {
  const [imgUrl, setImgUrl] = useState(null);
  const [detections, setDetections] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [naturalSize, setNaturalSize] = useState({ w: 1, h: 1 });
  const imgRef = useRef(null);

  async function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    setImgUrl(URL.createObjectURL(file));
    setDetections([]);
    setError(null);
    setLoading(true);

    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API_URL}/recognize`, { method: "POST", body: form });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      setDetections(data.detections);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function handleImgLoad() {
    setNaturalSize({
      w: imgRef.current.naturalWidth,
      h: imgRef.current.naturalHeight,
    });
  }

  const displayWidth = 600; // matches the CSS max-width below
  const scale = displayWidth / naturalSize.w;

  return (
    <div style={{ padding: 20, fontFamily: "sans-serif" }}>
      <h1>SKU Recognition</h1>
      <input type="file" accept="image/*" onChange={handleUpload} />
      {loading && <p>Analyzing...</p>}
      {error && <p style={{ color: "red" }}>{error}</p>}

      {imgUrl && (
        <div style={{ position: "relative", display: "inline-block", marginTop: 16 }}>
          <img
            ref={imgRef}
            src={imgUrl}
            onLoad={handleImgLoad}
            style={{ width: displayWidth, display: "block" }}
            alt="uploaded"
          />
          {detections.map((d, i) => {
            const [x1, y1, x2, y2] = d.bbox;
            return (
              <div
                key={i}
                style={{
                  position: "absolute",
                  left: x1 * scale,
                  top: y1 * scale,
                  width: (x2 - x1) * scale,
                  height: (y2 - y1) * scale,
                  border: "2px solid red",
                  boxSizing: "border-box",
                }}
              >
                <span style={{
                  position: "absolute", top: -20, left: 0,
                  background: "red", color: "white", fontSize: 12, padding: "1px 4px",
                }}>
                  {d.predictions[0][0]} ({d.predictions[0][1].toFixed(2)})
                </span>
              </div>
            );
          })}
        </div>
      )}

      {detections.map((d, i) => (
        <div key={i} style={{ marginTop: 8, padding: 8, border: "1px solid #ccc" }}>
          <b>Detection {i + 1}</b> — YOLO conf {d.yolo_conf.toFixed(2)}
          {d.is_unknown && <span style={{ color: "orange" }}> (low confidence)</span>}
          <div style={{ fontSize: 12, color: "#666" }}>
            Top-5: {d.predictions.slice(0, 5).map(p => `${p[0]} (${p[1].toFixed(2)})`).join(", ")}
          </div>
        </div>
      ))}
    </div>
  );
}
