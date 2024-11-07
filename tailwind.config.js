/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#FFE600',
          dark: '#FFD700',
          light: '#FFF44F'
        },
        secondary: {
          DEFAULT: '#2E2E38',
          dark: '#1D1D24',
          light: '#3F3F4E'
        },
        light: {
          bg: '#F8FAFC',
          sidebar: '#FFFFFF',
          hover: '#F1F5F9',
          border: '#E2E8F0',
          text: {
            primary: '#1E293B',
            secondary: '#64748B',
            accent: '#334155'
          }
        },
        dark: {
          bg: '#0F172A',
          sidebar: '#1E293B',
          hover: '#334155',
          border: '#1E293B',
          text: {
            primary: '#F8FAFC',
            secondary: '#94A3B8',
            accent: '#CBD5E1'
          }
        }
      },
      boxShadow: {
        'light-glow': '0 0 20px rgba(255, 230, 0, 0.15)',
        'message': '0 2px 8px rgba(0, 0, 0, 0.05)',
        'message-hover': '0 4px 12px rgba(0, 0, 0, 0.1)',
      },
      animation: {
        'float': 'float 3s ease-in-out infinite',
        'fadeIn': 'fadeIn 0.5s ease-out',
        'slideUp': 'slideUp 0.5s ease-out',
        'glow': 'glow 2s ease-in-out infinite',
        'spin-slow': 'spin 3s linear infinite',
        'bounce-slow': 'bounce 3s infinite',
        'pulse-subtle': 'pulseSubtle 2s infinite',
      },
      keyframes: {
        pulseSubtle: {
          '0%, 100%': { opacity: 1 },
          '50%': { opacity: 0.8 },
        }
      }
    },
  },
  plugins: [],
};