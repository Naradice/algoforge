import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-inter)", "ui-sans-serif", "system-ui", "-apple-system", "BlinkMacSystemFont", "sans-serif"],
      },
      colors: {
        // Material Design dark surface palette
        // These override Tailwind's default grays throughout the app
        gray: {
          50:  "#F0F6FC",
          100: "#E6EDF3",
          200: "#C9D1D9",
          300: "#B1BAC4",
          400: "#8B949E",
          500: "#6E7681",
          600: "#484F58",
          700: "#30363D",
          800: "#21262D",
          900: "#161B22",
          950: "#0D1117",
        },
        // Primary brand color — Material Blue
        brand: {
          50:  "#E8F4FD",
          100: "#C3E0FA",
          200: "#9DCFF7",
          300: "#71BBF4",
          400: "#58A6FF",
          500: "#2F81F7",
          600: "#1F6FD9",
          700: "#1558B0",
          800: "#0D3F87",
          900: "#082968",
        },
        // Semantic status colors
        success: {
          DEFAULT: "#3FB950",
          muted: "#1A3A22",
        },
        warning: {
          DEFAULT: "#D29922",
          muted: "#3A2E0A",
        },
        danger: {
          DEFAULT: "#F85149",
          muted: "#3A1414",
        },
      },
      boxShadow: {
        // Material Design elevation shadows
        "elevation-1": "0 1px 2px rgba(0,0,0,0.4), 0 1px 3px 1px rgba(0,0,0,0.2)",
        "elevation-2": "0 1px 2px rgba(0,0,0,0.4), 0 2px 6px 2px rgba(0,0,0,0.2)",
        "elevation-3": "0 4px 8px 3px rgba(0,0,0,0.2), 0 1px 3px rgba(0,0,0,0.4)",
        "elevation-4": "0 6px 10px 4px rgba(0,0,0,0.2), 0 2px 3px rgba(0,0,0,0.4)",
      },
      borderRadius: {
        "2xl": "1rem",
        "3xl": "1.5rem",
      },
    },
  },
  plugins: [],
};

export default config;
