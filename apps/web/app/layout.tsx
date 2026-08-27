import type { Metadata } from "next";
import { ClerkProvider, SignedIn, SignedOut, SignInButton, UserButton } from "@clerk/nextjs";
import "./globals.css";

export const metadata: Metadata = {
  title: "EditGPT",
  description: "Describe the change. Get the image.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <ClerkProvider>
      <html lang="en">
        <body className="min-h-screen bg-white text-neutral-900 antialiased dark:bg-neutral-950 dark:text-neutral-100">
          <header className="flex items-center justify-between border-b border-neutral-200 px-6 py-3 dark:border-neutral-800">
            <span className="font-semibold tracking-tight">EditGPT</span>
            <SignedOut>
              <SignInButton mode="modal">
                <button className="rounded-md border border-neutral-300 px-3 py-1.5 text-sm dark:border-neutral-700">
                  Sign in
                </button>
              </SignInButton>
            </SignedOut>
            <SignedIn>
              <UserButton />
            </SignedIn>
          </header>
          {children}
        </body>
      </html>
    </ClerkProvider>
  );
}
