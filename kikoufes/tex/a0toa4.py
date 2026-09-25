from pypdf import PdfReader, PdfWriter, Transformation


def split_a0_to_a4_16pages(input_pdf_path, output_pdf_path):
    reader = PdfReader(input_pdf_path)
    writer = PdfWriter()

    # 最初のページを取得
    page = reader.pages[0]

    # A0の元のサイズを取得 (ポイント単位: 1ポイント = 1/72インチ)
    # A0: 約 841mm × 1189mm -> 約 2384pt × 3370pt
    original_width = page.mediabox.width
    original_height = page.mediabox.height

    # 4×4の16枚に分割するため、1枚あたりのサイズを計算
    # (A0を4分割すると、ちょうどA4に非常に近いサイズ・比率になります)
    grid_cols = 4  # 横に4分割
    grid_rows = 4  # 縦に4分割

    target_width = original_width / grid_cols
    target_height = original_height / grid_rows

    # 左上から右下へ向かって、1マスずつ切り出して新しいページにする
    # PDFの座標系は「左下が (0, 0)」になる点に注意
    for row in range(grid_rows):
        for col in range(grid_cols):
            # 新しいA4サイズ相当のページを作成
            new_page = writer.add_blank_page(
                width=target_width, height=target_height
            )

            # 元のA0ページから、該当するエリアが(0,0)の位置に来るように平移（シフト）させる
            # 縦は上から順に処理するため、全体の高さから差し引く
            tx = -col * target_width
            ty = -(grid_rows - 1 - row) * target_height

            transform = Transformation().translate(tx, ty)

            # 重ね合わせる（切り出しの実行）
            new_page.merge_page(page)
            new_page.add_transformation(transform)

    # 16ページになったPDFを保存
    with open(output_pdf_path, "wb") as f_out:
        writer.write(f_out)

    print(
        f"成功しました！ {output_pdf_path} に16ページのPDFを出力しました。"
    )


# 実行部分
if __name__ == "__main__":
    # 実際のファイル名に合わせて変更してください
    input_file = "poster.pdf"
    output_file = "output_a4_16pages.pdf"

    split_a0_to_a4_16pages(input_file, output_file)
