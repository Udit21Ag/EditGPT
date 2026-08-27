import { Show, SignInButton } from "@clerk/nextjs";
import { EditWorkspace } from "@/components/EditWorkspace";

export default function Home() {
  return (
    <main className="min-h-screen">
      <Show
        when="signed-in"
        fallback={
          <div className="mx-auto flex min-h-[70vh] max-w-2xl flex-col justify-center gap-4 px-6">
            <h1 className="text-4xl font-semibold tracking-tight">EditGPT</h1>
            <p className="text-neutral-600 dark:text-neutral-400">
              Describe the change. Get the image.
            </p>
            <SignInButton mode="modal">
              <button className="w-fit rounded-md bg-neutral-900 px-4 py-2 text-sm font-medium text-white dark:bg-white dark:text-neutral-900">
                Sign in to start
              </button>
            </SignInButton>
          </div>
        }
      >
        <EditWorkspace />
      </Show>
    </main>
  );
}
