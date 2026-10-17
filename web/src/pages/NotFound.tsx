import { Link } from "react-router-dom";
import { useSelection } from "@/lib/selection";

export default function NotFound() {
  const { search } = useSelection();
  return (
    <div className="flex flex-col items-center justify-center py-24 text-center">
      <div className="font-mono text-[12px] uppercase tracking-[0.2em] text-fg-muted">404</div>
      <h1 className="mt-2 text-[24px] font-semibold text-fg">Nothing trades here</h1>
      <p className="mt-2 max-w-md text-[13px] text-fg-muted">This page does not exist. Head back to the run brief.</p>
      <Link to={{ pathname: "/", search }} className="focus-ring mt-6 rounded-lg bg-accent px-4 py-2 text-[13px] font-semibold text-accent-ink hover:brightness-110">
        Back to Command
      </Link>
    </div>
  );
}
