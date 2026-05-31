/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        border: "var(--border)",
        ring: "var(--ring)",
        background: "var(--background)",
        foreground: "var(--foreground)",
        // add more variables here if needed
      },
      outline: {
        ring: "2px solid var(--ring)",
      },
    },
  },
  plugins: [],
  darkMode: "class",
};
