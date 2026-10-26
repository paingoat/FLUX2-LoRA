import { useEffect, useState } from "react";

function sampleFromLocation(samples) {
  const requested = new URLSearchParams(window.location.search).get("sample");
  const found = samples.findIndex((sample) => sample.index === requested);
  return found >= 0 ? found : 0;
}

export default function App() {
  const [samples, setSamples] = useState(null);
  const [index, setIndex] = useState(0);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    fetch("/api/samples")
      .then((response) => {
        if (!response.ok) throw new Error("Could not load the dataset.");
        return response.json();
      })
      .then((data) => {
        if (cancelled) return;
        setSamples(data);
        setIndex(sampleFromLocation(data));
      })
      .catch((loadError) => {
        if (!cancelled) setError(loadError.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!samples?.length) return undefined;
    const sample = samples[index];
    const url = new URL(window.location.href);
    url.searchParams.set("sample", sample.index);
    window.history.replaceState(null, "", url);

    function onKeyDown(event) {
      if (event.key === "ArrowLeft") {
        setIndex((current) => Math.max(0, current - 1));
      }
      if (event.key === "ArrowRight") {
        setIndex((current) => Math.min(samples.length - 1, current + 1));
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [samples, index]);

  if (error) {
    return <p className="status">{error}</p>;
  }
  if (!samples) {
    return <p className="status">Loading</p>;
  }
  if (samples.length === 0) {
    return <p className="status">No samples found.</p>;
  }

  const sample = samples[index];
  return (
    <main>
      <p className="prompt">{sample.prompt}</p>
      <div className="meta">
        <button type="button" onClick={() => setIndex((current) => current - 1)} disabled={index === 0}>
          Previous
        </button>
        <span>
          {index + 1} / {samples.length}
        </span>
        <span className="sample-id">{sample.sampleId}</span>
        <button
          type="button"
          onClick={() => setIndex((current) => current + 1)}
          disabled={index === samples.length - 1}
        >
          Next
        </button>
      </div>
      <section className="pair">
        <figure>
          <figcaption>Input</figcaption>
          <img src={sample.reference} alt={`Input ${sample.index}`} />
        </figure>
        <figure>
          <figcaption>GT</figcaption>
          <img src={sample.target} alt={`Ground truth ${sample.index}`} />
        </figure>
      </section>
    </main>
  );
}
