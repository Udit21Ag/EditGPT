import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";

/**
 * Which pages need a session.
 *
 * An allowlist of *public* routes rather than a list of protected ones: a new page added
 * six months from now is protected by default, whereas a list of protected routes silently
 * leaves anything nobody remembered to add wide open.
 */
const isPublic = createRouteMatcher(["/", "/sign-in(.*)", "/sign-up(.*)"]);

export default clerkMiddleware(async (auth, request) => {
  if (!isPublic(request)) {
    await auth.protect();
  }
});

export const config = {
  matcher: [
    // Everything except Next's internals and static files, plus every API route.
    "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)",
    "/(api|trpc)(.*)",
  ],
};
