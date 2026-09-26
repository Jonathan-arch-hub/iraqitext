"""Flet GUI demo for iraqitext (optional, requires `pip install flet`).

Not part of the library and not part of the test suite — run it manually:

    pip install flet
    python examples/flet_demo.py
"""
import importlib

import flet as ft

# استيراد المكتبة كـ module لإمكانية إعادة تحميله
try:
    import iraqitext
    it = iraqitext.IraqiText()
except ImportError:
    # كلاس افتراضي للتجربة في حال عدم وجود المكتبة
    class FakeIraqiText:
        def to_fusha(self, text): return f"فصحى: {text}"
        def to_iraqi(self, text): return f"عراقي: {text}"
    iraqitext = None
    it = FakeIraqiText()


def main(page: ft.Page):
    global it
    page.title = "مترجم IraqiText"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 20

    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.MainAxisAlignment.CENTER

    fusha_output = ft.Text(value="---", size=18, selectable=True)
    iraqi_output = ft.Text(value="---", size=18, selectable=True)

    # دالة إعادة تحميل القاموس والمكتبة قسراً من القرص الصلب
    def on_reload_dict(e):
        global it
        try:
            if iraqitext:
                importlib.reload(iraqitext)  # إعادة قراءة ملفات المكتبة والـ JSON من جديد
                it = iraqitext.IraqiText()

            page.snack_bar = ft.SnackBar(ft.Text("تمت إعادة تحميل القاموس والمكتبة من القرص بنجاح!"))
            page.snack_bar.open = True
        except Exception as err:
            page.snack_bar = ft.SnackBar(ft.Text(f"حدث خطأ أثناء التحديث: {err}"))
            page.snack_bar.open = True
        page.update()

    # دالة الترجمة
    def on_translate(e):
        val = text_input.value.strip() if text_input.value else ""
        if val:
            fusha_output.value = it.to_fusha(val)
            iraqi_output.value = it.to_iraqi(val)
        else:
            fusha_output.value = "يرجى كتابة نص أولاً!"
            iraqi_output.value = "يرجى كتابة نص أولاً!"
        page.update()

    text_input = ft.TextField(
        label="أدخل الكود / النص هنا",
        width=450,
        text_align=ft.TextAlign.RIGHT,
        on_submit=on_translate,
    )

    translate_btn = ft.FilledButton(
        "ترجمة",
        on_click=on_translate,
        width=200,
    )

    reload_btn = ft.OutlinedButton(
        "تحديث القاموس",
        on_click=on_reload_dict,
        width=200,
    )

    results_card = ft.Card(
        content=ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("إلى الفصحى:", weight=ft.FontWeight.BOLD, size=15),
                    fusha_output,
                    ft.Divider(),
                    ft.Text("إلى العراقي:", weight=ft.FontWeight.BOLD, size=15),
                    iraqi_output,
                ],
                spacing=10,
            ),
            padding=20,
            width=450,
        )
    )

    buttons_row = ft.Row(
        controls=[translate_btn, reload_btn],
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=15,
    )

    page.add(
        ft.Column(
            controls=[
                ft.Text("مترجم IraqiText", size=26, weight=ft.FontWeight.BOLD),
                text_input,
                buttons_row,
                results_card,
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=20,
        )
    )


if __name__ == "__main__":
    ft.run(main)
