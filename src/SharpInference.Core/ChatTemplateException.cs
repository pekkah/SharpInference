namespace SharpInference.Core;

/// <summary>
/// Raised when a chat template deliberately rejects the messages it was given, via Jinja's
/// <c>raise_exception(...)</c>. Several model families guard their own conversation shape this way
/// — Mistral's v3 template, for instance, refuses a history whose roles don't strictly alternate
/// user/assistant after the optional system message.
///
/// <para>This is distinct from an engine fault: the template ran correctly and the *input* was
/// invalid, so API endpoints map it to HTTP 400 with the template's own message rather than
/// letting it surface as an opaque 500. Any other exception escaping the renderer still means a
/// defect on our side and is left to fail loudly.</para>
///
/// <para>Derives from <see cref="InvalidOperationException"/> so existing callers that catch that
/// type keep working unchanged.</para>
/// </summary>
public sealed class ChatTemplateException(string message) : InvalidOperationException(message);
