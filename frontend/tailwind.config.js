/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        dark: {
          900: '#0d1117',
          800: '#161b22',
          700: '#1c2333',
          600: '#2d333b',
          500: '#444c56',
        },
      },
    },
  },
  plugins: [],
}

