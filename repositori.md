## Link Repositori
[Link ke Repositori](URL_REPOSITORI_GITHUB)

## Hasil Eksperimen
### Absolute Difference
- **Ukuran Kernel**: Menggunakan kernel (5,5) menghasilkan gambar dengan lebih sedikit noise dibandingkan kernel (7,7).
- **Bentuk Kernel**: Kernel elliptical lebih baik dalam mempertahankan bentuk objek dibandingkan kernel rectangular.

### Mixture of Gaussian 2
- **Ukuran Kernel**: Kernel yang lebih besar menghasilkan tepi objek yang lebih halus, tetapi dapat menghilangkan detail kecil.
- **Bentuk Kernel**: Bentuk kernel cross tidak seefektif bentuk kernel rectangular dalam mengisi lubang kecil pada objek.

### KNN
- **Variasi Operasi Morfologi**: Menambahkan operasi `cv.MORPH_OPEN` sebelum `cv.MORPH_CLOSE` membantu dalam menghilangkan noise kecil sebelum dilasi, memberikan hasil segmentasi yang lebih bersih.