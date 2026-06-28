export function Field({ label, children }) {
  return (
    <div className="Field">
      <label>{label}</label>
      {children}
    </div>
  );
}

