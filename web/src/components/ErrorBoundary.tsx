import { Component, type ErrorInfo, type ReactNode } from "react";

/** Contains a rendering error to the page it happened in; the shell and navigation keep working. */
export class ErrorBoundary extends Component<{ children: ReactNode; resetKey?: string }, { error: Error | null }> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("View crashed:", error, info.componentStack);
  }

  componentDidUpdate(prev: { resetKey?: string }) {
    if (prev.resetKey !== this.props.resetKey && this.state.error) this.setState({ error: null });
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="rounded-xl border border-critical/30 bg-critical/[0.05] px-5 py-6">
        <div className="text-[14px] font-semibold text-fg">This view hit an error</div>
        <div className="mt-1 font-mono text-[12px] text-fg-muted">{this.state.error.message}</div>
        <button
          onClick={() => this.setState({ error: null })}
          className="focus-ring mt-4 rounded-lg border border-ink-600 px-3 py-1.5 text-[12.5px] text-fg-2 hover:bg-tint/5"
        >
          Try again
        </button>
      </div>
    );
  }
}
