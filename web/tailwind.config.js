/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "rgb(var(--c-bg) / <alpha-value>)",
          900: "rgb(var(--c-bg-2) / <alpha-value>)",
          850: "rgb(var(--c-panel) / <alpha-value>)",
          800: "rgb(var(--c-panel-2) / <alpha-value>)",
          700: "rgb(var(--c-line) / <alpha-value>)",
          600: "rgb(var(--c-line-2) / <alpha-value>)",
        },
        fg: {
          DEFAULT: "rgb(var(--c-text) / <alpha-value>)",
          2: "rgb(var(--c-text-2) / <alpha-value>)",
          muted: "rgb(var(--c-muted) / <alpha-value>)",
        },
        accent: {
          DEFAULT: "rgb(var(--c-accent) / <alpha-value>)",
          soft: "var(--accent-soft)",
          ink: "rgb(var(--c-accent-ink) / <alpha-value>)",
        },
        good: "rgb(var(--c-good) / <alpha-value>)",
        warn: "rgb(var(--c-warn) / <alpha-value>)",
        serious: "rgb(var(--c-serious) / <alpha-value>)",
        critical: "rgb(var(--c-critical) / <alpha-value>)",
        tint: "rgb(var(--c-tint) / <alpha-value>)",
      },
      fontFamily: {
        sans: ['"IBM Plex Sans"', "system-ui", "-apple-system", '"Segoe UI"', "sans-serif"],
        mono: ['"IBM Plex Mono"', "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      boxShadow: {
        panel: "0 1px 0 0 rgb(var(--c-tint) / 0.03) inset, 0 12px 32px -18px var(--shadow-drop)",
        glow: "0 0 0 1px rgb(var(--c-accent) / 0.35), 0 8px 30px -10px rgb(var(--c-accent) / 0.35)",
      },
      keyframes: {
        marquee: { "0%": { transform: "translateX(0)" }, "100%": { transform: "translateX(-50%)" } },
        pulseDot: { "0%,100%": { opacity: 1 }, "50%": { opacity: 0.35 } },
        shimmer: { "0%": { backgroundPosition: "-400px 0" }, "100%": { backgroundPosition: "400px 0" } },
      },
      animation: {
        marquee: "marquee 60s linear infinite",
        pulseDot: "pulseDot 1.8s ease-in-out infinite",
        shimmer: "shimmer 1.4s linear infinite",
      },
    },
  },
  plugins: [],
};
