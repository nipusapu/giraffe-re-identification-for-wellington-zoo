import type { Metadata } from 'next';
import { Geist, Geist_Mono, Ubuntu } from 'next/font/google';
import './globals.css';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

const ubuntu = Ubuntu({
  weight: '400',
  subsets: ['latin'],
  variable: '--font-ubuntu',
});

export const metadata: Metadata = {
  title: 'Giraffe Re-Identification',
  description: 'Web application for identifying giraffes in Wellington Zoo',
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  const year = new Date().getFullYear();

  return (
    <html lang="en" className={ubuntu.className}>
      {/* Keep the original body wrapper to preserve your responsive height behavior */}
      <body className="layout-background">
        {children}

        {/* Fixed footer overlays the background; doesn't affect page height */}
        <footer className="fixed bottom-0 left-0 right-0 z-50 text-center text-[10px] sm:text-xs text-white/80 py-2 bg-black/30 backdrop-blur">
          © {year} All right resieved to Nipuna
        </footer>
      </body>
    </html>
  );
}
