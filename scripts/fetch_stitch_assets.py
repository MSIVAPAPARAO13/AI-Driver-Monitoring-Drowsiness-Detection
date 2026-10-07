import urllib.request
import os
from pathlib import Path

screens = [
    {
        'id': 'offline_video_analysis',
        'img_url': 'https://lh3.googleusercontent.com/aida/AEtjO1UfANo6IJfmzTXDXHgY_c0FjGc0oGNmuQ2yl9dWZtbJaKUU94Tqf3EUgTyYOyAHNwBmlLmBQGFxwLg0487lO0GXF0lB_s4R2Nv8nD9pnVrqfid9n4ctiBUt9kpByL4-AN0Ofb1pX8TBjbdU0rrZ7Q3iBz1ugzD_hoxSVx2UM8Gaq6_Fs-azyhuJduihSZf8GxYjvy0bsCBW71qUpYqFMPLrtyhDrY5GQm503LN8befZfMCbF-aIZbAncg',
        'html_url': 'https://contribution.usercontent.google.com/download?c=CgthaWRhX2NvZGVmeBJ7Eh1hcHBfY29tcGFuaW9uX2dlbmVyYXRlZF9maWxlcxpaCiVodG1sXzAwMDY1ZDJhYjBjMjM4MjEwOTEwNGRhZDM1MTI3YTQ0EgsSBxCZq6rsowkYAZIBIwoKcHJvamVjdF9pZBIVQhM1Njg2NDQ0MjI2NzI3NzE0NjY2&filename=&opi=89354086'
    },
    {
        'id': 'continual_learning_workspace',
        'img_url': 'https://lh3.googleusercontent.com/aida/AEtjO1VBtyNFJnTSBZWoZMOXQJIGdun5CoCN9dtj0INHJc7Zghmk_GeKT7AMcwkM39O3lC31rJVtFDViSxulbBk74lkwMZ8dCqB6vgR4gyBwZMfyWXsMHx5CeYuEHhWuLLZ-97sf3Bm00ZELZ_PDPjSg_ibvzMg4S59elvE1DCPrXW934AeF84_OTXU3X9T37vrbS6EDdb732B35hVSqrlDM22ajCjBk0vY2q4ixyY2aoCsdiHsyH1S71qwOfvU',
        'html_url': 'https://contribution.usercontent.google.com/download?c=CgthaWRhX2NvZGVmeBJ7Eh1hcHBfY29tcGFuaW9uX2dlbmVyYXRlZF9maWxlcxpaCiVodG1sXzAwMDY1ZDJhYjE0OWYxNTIwMmRlYTY2YTc1MGJmMGNiEgsSBxCZq6rsowkYAZIBIwoKcHJvamVjdF9pZBIVQhM1Njg2NDQ0MjI2NzI3NzE0NjY2&filename=&opi=89354086'
    },
    {
        'id': 'architecture_privacy',
        'img_url': 'https://lh3.googleusercontent.com/aida/AEtjO1VOHVKnZZnwQYd792KxMRF5wWsq5i6btHBReyECozOX2ASRpWc0zblrjG8xu6YDl2CoeVlb7doP1lBVIRp66YBobcsStIHjsLN61zQu5K05k3aa2gqmHZNVVelYVIcgSxE85qWIWQdH7Ia6YNX7uddwfsSen1PY9G7hOrmgfYxiBTWUzartfn4UAJMlzJoQJSvsdw7xXH0FoEjZ16oUkShAjBgonUh_rAUt4PEAehdaW5bd5z5L_7OoqBM',
        'html_url': 'https://contribution.usercontent.google.com/download?c=CgthaWRhX2NvZGVmeBJ7Eh1hcHBfY29tcGFuaW9uX2dlbmVyYXRlZF9maWxlcxpaCiVodG1sXzAwMDY1ZDJhYjE0MjlkYzkwMzM4NWE2MDI4MDgyNjBiEgsSBxCZq6rsowkYAZIBIwoKcHJvamVjdF9pZBIVQhM1Njg2NDQ0MjI2NzI3NzE0NjY2&filename=&opi=89354086'
    },
    {
        'id': 'live_monitor',
        'img_url': 'https://lh3.googleusercontent.com/aida/AEtjO1VTLMcZmE6aap5UKRj8x4Wrk74A0wsuDwl9X5u0LRo9b8LtvcFLl5DAv7nkKu6rtdsV5mbzbmDKji4V3at6WI64mIFMRujF1YxstL6Efd0sIhe12vF9yO_uGiBveLz7612ZaGDC1aC6-g9EGVrEUBNl1N2hN7KIHPAdiBxs9Dr6u3x3DlSW0xtjx8BSv-2Lw144A0gwxjA0-ulbWtg3ocQGoR7KR4xrH7oz0ILkLk1elG4SFOMrskIorN4',
        'html_url': 'https://contribution.usercontent.google.com/download?c=CgthaWRhX2NvZGVmeBJ7Eh1hcHBfY29tcGFuaW9uX2dlbmVyYXRlZF9maWxlcxpaCiVodG1sXzAwMDY1ZDJhYjFjYjU3YzQwNTAzYzFmMTcwMTU2NmJlEgsSBxCZq6rsowkYAZIBIwoKcHJvamVjdF9pZBIVQhM1Njg2NDQ0MjI2NzI3NzE0NjY2&filename=&opi=89354086'
    },
    {
        'id': 'model_performance_benchmarks',
        'img_url': 'https://lh3.googleusercontent.com/aida/AEtjO1XpiCzmffdcS4CH33gJCMMSXT5uI6dR7lcAl0wQiQKvDvGUsgxpiNeB3l05ea5TJkSnQIWfRQDfSCtDNcOHBCCsL-jlji5ToFmp4s8UTk92cvOljrKh9TjB-hsSGSyZnbAZUkXP-h3aA2U3gRNT2oGelU07eQbpZa62KFCQrduRK6jD4SHUqbNc13uK88yyXJn9RtVhDtysNY6Ykgcurl7JvoiXYihbySnuDWmqr1CAI9fbVdBFzC4wmm0',
        'html_url': 'https://contribution.usercontent.google.com/download?c=CgthaWRhX2NvZGVmeBJ7Eh1hcHBfY29tcGFuaW9uX2dlbmVyYXRlZF9maWxlcxpaCiVodG1sXzAwMDY1ZDJhYjE2OTJkYWMwMjJkNjdmMGU3MjczNDg4EgsSBxCZq6rsowkYAZIBIwoKcHJvamVjdF9pZBIVQhM1Njg2NDQ0MjI2NzI3NzE0NjY2&filename=&opi=89354086'
    }
]

out_dir = Path('app/assets/stitch')
out_dir.mkdir(parents=True, exist_ok=True)
html_dir = Path('app/assets/stitch/html')
html_dir.mkdir(parents=True, exist_ok=True)

headers = {'User-Agent': 'Mozilla/5.0'}

for s in screens:
    # Image
    img_path = out_dir / f"{s['id']}.webp"
    req = urllib.request.Request(s['img_url'], headers=headers)
    with urllib.request.urlopen(req) as resp, open(img_path, 'wb') as f:
        f.write(resp.read())
    print(f"Saved image: {img_path} ({os.path.getsize(img_path)} bytes)")

    # HTML
    html_path = html_dir / f"{s['id']}.html"
    req = urllib.request.Request(s['html_url'], headers=headers)
    with urllib.request.urlopen(req) as resp, open(html_path, 'wb') as f:
        f.write(resp.read())
    print(f"Saved html: {html_path} ({os.path.getsize(html_path)} bytes)")
