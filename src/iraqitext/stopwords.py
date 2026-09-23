"""Small built-in Arabic stopword list.

Enough for :meth:`IraqiText.keywords` without pulling in a heavy stopword
corpus. Supply your own set via ``IraqiText.keywords(..., stopwords=...)``
when you need something bigger or dialect-specific.
"""

STOPWORDS = frozenset(
    """
    و في من على عن إلى حتى منذ نحو خلال بعد قبل بين عند مع كل بعض
    هذا هذه ذلك تلك هؤلاء الذي التي الذين اللذان اللتان
    أن إن أنا نحن أنت أنتم هو هي هم هن كانت كان كانوا يكون تكون
    أين متى لماذا كيف ماذا هل ما لا لم لن ثم إذا لو لكن إلا غير سوى
    حسب فقط أي يا أيها وأيضا أيضاً كذلك يوجد يمكن يجب ينبغي ربما
    """.split()
)