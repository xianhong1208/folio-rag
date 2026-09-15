/** @type {import('tailwindcss').Config} */
// Retrieval Terminal — tokens are CSS vars defined in src/index.css (dark-first + light).
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        void: 'var(--body-bg)',
        surface: 'var(--surface)',
        elevated: 'var(--elevated)',
        input: 'var(--input-bg)',
        line: 'var(--hairline)',
        'line-soft': 'var(--hairline-soft)',
        ink: 'var(--ink)',
        'ink-muted': 'var(--ink-muted)',
        'ink-subtle': 'var(--ink-subtle)',
        cyan: 'var(--cyber)',
        'cyan-hi': 'var(--cyber-hi)',
        amber: 'var(--signal)',
        green: 'var(--matrix)',
        red: 'var(--alert)',
      },
      fontFamily: {
        display: ['"Chakra Petch"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'Menlo', 'monospace'],
      },
    },
  },
  plugins: [],
}
