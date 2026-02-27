import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DEFault: Detection & Explain Fault",
  description:
    "DEFault: Detect, categorize, and explain faults in deep neural networks using hierarchical and explainable AI classification.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-screen antialiased" style={{ background: "var(--bg-base)", color: "var(--text-primary)" }}>
        {/* Skip to main content for keyboard/screen-reader users */}
        <a href="#panel-pipeline" className="skip-link">Skip to analysis pipeline</a>
        {children}
      </body>
    </html>
  );
}
