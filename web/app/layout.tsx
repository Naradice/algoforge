import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Sidebar } from "@/components/sidebar";
import { ToastProvider } from "@/lib/toast";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "AlgoForge",
  description: "Algorithmic trading platform — Strategy · Model · Data",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="flex h-screen overflow-hidden">
        <ToastProvider>
          <Sidebar />
          <main className="flex-1 overflow-y-auto bg-gray-950 px-8 py-7">{children}</main>
        </ToastProvider>
      </body>
    </html>
  );
}
