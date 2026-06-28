import { useMemo, useState } from "react";
import { Field } from "./Field";

function coerceValue(field, raw) {
  if (field.type === "number") {
    if (raw === "" || raw === null || raw === undefined) return null;
    const v = Number(raw);
    return Number.isFinite(v) ? v : null;
  }
  if (field.type === "boolean") return Boolean(raw);
  return raw;
}

export function DynamicJobForm({ job, initial = {}, onSubmit, submitLabel = "Start", extraButtons = null }) {
  const defaults = useMemo(() => {
    const obj = {};
    for (const f of job.args || []) {
      obj[f.key] = f.default ?? "";
    }
    return obj;
  }, [job]);

  const [values, setValues] = useState({ ...defaults, ...initial });
  const [error, setError] = useState("");

  const submit = async () => {
    setError("");
    // validate required
    for (const f of job.args || []) {
      if (!f.required) continue;
      const v = values[f.key];
      if (v === null || v === undefined || String(v).trim() === "") {
        setError(`Missing required field: ${f.label}`);
        return;
      }
    }
    const args = {};
    for (const f of job.args || []) {
      let v = values[f.key];
      if (f.type === "boolean") {
        v = Boolean(v);
      } else if (f.type === "number") {
        v = coerceValue(f, v);
      } else {
        v = String(v ?? "");
      }
      // send null/empty as null for optional fields
      if (!f.required && (v === "" || v === null)) {
        continue;
      }
      args[f.key] = v;
    }
    await onSubmit(args);
  };

  return (
    <div>
      {error ? <div className="Status">Error: {error}</div> : null}
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {(job.args || []).map((f) => (
          <Field key={f.key} label={`${f.label}${f.required ? " *" : ""}`}>
            {f.choices && f.choices.length ? (
              <select
                className="Select"
                value={values[f.key] ?? ""}
                onChange={(e) => setValues((p) => ({ ...p, [f.key]: e.target.value }))}
              >
                {f.required ? null : <option value="">(none)</option>}
                {f.choices.map((c) => (
                  <option key={String(c)} value={String(c)}>
                    {String(c)}
                  </option>
                ))}
              </select>
            ) : f.type === "boolean" ? (
              <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <input
                  type="checkbox"
                  checked={Boolean(values[f.key])}
                  onChange={(e) => setValues((p) => ({ ...p, [f.key]: e.target.checked }))}
                />
                <span className="Status">{f.help || ""}</span>
              </label>
            ) : (
              <input
                className="Input"
                type={f.type === "number" ? "number" : "text"}
                value={values[f.key] ?? ""}
                placeholder={f.placeholder || ""}
                onChange={(e) => setValues((p) => ({ ...p, [f.key]: e.target.value }))}
              />
            )}
            {f.help && f.type !== "boolean" ? <div className="Status">{f.help}</div> : null}
          </Field>
        ))}
      </div>
      <div className="Row" style={{ marginTop: 10 }}>
        <button className="Button" onClick={submit}>
          {submitLabel}
        </button>
        {extraButtons}
      </div>
    </div>
  );
}

