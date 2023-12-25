# MachineLearning_Optimizer_ContinualLearning_TestProduction
# Tác Giả
Nguyễn Tấn Thành
# Giới thiệu
Tôi xin giới thiệu báo cáo của mình về các phương pháp tối ưu trong Deep Learning. Báo cáo tập trung mô tả chi tiết về các phương pháp tối ưu đang được sử dụng rộng rãi trong cộng đồng Deep Learning ngày nay. Bắt đầu từ những lý thuyết và tư tưởng cơ bản của gradient descent, chúng tôi giải thích ưu và nhược điểm của từng phương pháp và đi sâu vào khía cạnh toán học của chúng.

Gradient descent, mặc dù có nhược điểm, vẫn là phương pháp dễ hiểu và phổ biến nhất. Tính đặc trưng của gradient descent là khó vượt qua điểm cực tiểu và khả năng không điều chỉnh learning rate cho từng tham số. Điều này dẫn đến sự xuất hiện của nhiều phương pháp tối ưu sau này, như momentum, NAG, AdaGrad, Adadelta, RMSProp và Adam, nhằm khắc phục nhược điểm của Gradient Descent. Các phương pháp này được thiết kế để tối ưu hóa quá trình huấn luyện mô hình.

Continual Learning (CL) đóng vai trò quan trọng trong lĩnh vực Machine Learning (ML), đòi hỏi khả năng liên tục học từ dữ liệu mới mà không bỏ qua kiến thức đã được học từ dữ liệu cũ. Trong quá trình triển khai giải pháp học máy để giải quyết bài toán, sự đối mặt với dữ liệu mới và thay đổi liên tục là một thực tế phổ biến và quan trọng.

Test Production là quá trình quan trọng, trong đó chúng tôi tạo và triển khai các tập kiểm thử để đánh giá hiệu suất của mô hình.

Chúng tôi cũng đã thực nghiệm việc lập trình lại các thí nghiệm và công bố mã nguồn mở từ các bài báo để cung cấp tài liệu tham khảo cho độc giả. Báo cáo chi tiết có sẵn trong tệp PDF, và mã nguồn có thể được xem trong các tệp Jupyter tương ứng.

# Bốn giai đoạn của việc Continual Learning
Giai đoạn 1: Đào tạo lại thủ công:
Các mô hình chỉ được đào tạo lại khi đáp ứng hai điều kiện: (1) hiệu suất của mô hình đã suy giảm đến mức hiện tại nó gây hại nhiều hơn là có lợi, (2) nhóm của bạn có thời gian để cập nhật mô hình.

Giai đoạn 2: Đào tạo lại không trạng thái tự động theo lịch trình cố định
Giai đoạn này thường xảy ra khi các mô hình chính của một miền đã được phát triển và do đó ưu tiên của bạn không còn là tạo các mô hình mới mà là duy trì và cải thiện các mô hình hiện có. Ở lại giai đoạn 1 đã trở thành một nỗi đau quá lớn không thể bỏ qua.
Tần suất đào tạo lại ở giai đoạn này thường dựa trên "cảm giác ruột thịt".
giai đoạn 1 và giai đoạn 2 thường là một tập lệnh do ai đó viết để chạy quá trình đào tạo lại không trạng thái theo định kỳ. Việc viết tập lệnh này có thể rất dễ hoặc rất khó tùy thuộc vào số lượng phần phụ thuộc cần được phối hợp để đào tạo lại một mô hình.

Giai đoạn 3: Đào tạo trạng thái tự động theo lịch trình cố định
Để đạt được điều này, bạn cần phải cấu hình lại tập lệnh của mình và cách theo dõi dòng dữ liệu cũng như mô hình của bạn. Một ví dụ về phiên bản dòng dõi mô hình đơn giản:
V1 và V2 là hai kiến ​​trúc mô hình khác nhau cho cùng một vấn đề.
V1.2 so với V2.3 có nghĩa là kiến ​​trúc mô hình V1 đang ở lần lặp thứ 2 của quá trình đào tạo lại không trạng thái hoàn toàn và V2 đang ở lần lặp thứ 3.

Giai đoạn 4: Continual learning
Trong giai đoạn này, phần lịch trình cố định của các giai đoạn trước được thay thế bằng một số cơ chế kích hoạt đào tạo lại. Các tác nhân kích hoạt có thể là:
Dựa trên thời gian
Dựa trên hiệu suất: ví dụ: hiệu suất đã giảm xuống dưới x%.
Không phải lúc nào bạn cũng có thể đo lường trực tiếp độ chính xác trong quá trình sản xuất, vì vậy bạn có thể cần sử dụng một số proxy yếu hơn.

# Testing models in Production
- chiến lực Testing models in Production
  
- Shadow Deployment: đề xuất triển khai một mô hình mới (gọi là "challenger model") cùng với mô hình hiện tại (gọi là "champion model"). Mỗi khi có một yêu cầu mới được gửi đến hệ thống, cả hai mô hình đều nhận yêu cầu đó. Tuy nhiên, chỉ kết quả dự đoán từ mô hình hiện tại (champion) được phục vụ cho người dùng.
Mục tiêu chính ở đây là thu thập dự đoán từ cả hai mô hình, sau đó ghi log lại những dự đoán này để có thể so sánh chúng sau này. Bạn không trực tiếp phục vụ dự đoán từ mô hình mới cho người dùng, nhưng thay vào đó, bạn sử dụng mô hình hiện tại để đảm bảo tính ổn định của hệ thống.
Ưu điểm:
An toàn: Đây là cách triển khai an toàn nhất vì dù có lỗi trong mô hình mới, dự đoán từ nó sẽ không được phục vụ cho người dùng.
Đơn giản: Chiến lược này có khá đơn giản về mặt khái niệm. Bạn chỉ cần triển khai mô hình thách thức song song với mô hình hiện tại.
Tích luỹ dữ liệu nhanh chóng: Vì tất cả các mô hình đều nhận toàn bộ lưu lượng, bạn sẽ thu thập đủ dữ liệu để đạt được tính chất thống kê nhanh chóng hơn so với các chiến lược khác
Nhược điểm:
Không thích hợp cho mô hình tư vấn (recommender models): Không thể sử dụng khi đo hiệu suất của mô hình phụ thuộc vào việc quan sát cách người dùng tương tác với các dự đoán. Ví dụ, không phục vụ các dự đoán từ mô hình tư vấn "shadow", do đó bạn không biết liệu người dùng có thể nhấp vào chúng hay không.
Chi phí cao: Phương pháp này tăng gấp đôi số lượng dự đoán và do đó tăng chi phí tính toán.
Xử lý các trường hợp đặc biệt: Nếu triển khai thực hiện thông qua các chế độ dự đoán trực tuyến, bạn phải xử lý các trường hợp đặc biệt như mô hình "shadow" mất nhiều thời gian hơn để phục vụ một dự đoán so với mô hình chính, hoặc nếu mô hình "shadow" gặp lỗi. Câu hỏi đặt ra là liệu mô hình chính cũng nên gặp lỗi hay không khi mô hình "shadow" gặp sự cố.

- A/B Testing:Triển khai mô hình thách thức (model B) cùng với mô hình hiện tại (model A) và chuyển một phần tráfic đến mô hình thách thức. Dự đoán từ mô hình thách thức được hiển thị cho người dùng. Sử dụng giám sát và phân tích dự đoán trên cả hai mô hình để xác định xem hiệu suất của mô hình thách thức có độ chênh lệch thống kê so với mô hình hiện tại hay không.
Ưu điểm:
Quan sát phản ứng của người dùng: Vì dự đoán được phục vụ cho người dùng, chiến lược này cho phép bạn thu thập cách người dùng phản ứng với các mô hình khác nhau.
Dễ triển khai và giá rẻ: Phương pháp này đơn giản để hiểu và triển khai, và chi phí thấp vì mỗi yêu cầu chỉ có một dự đoán.
Khả năng mở rộng: Bạn có thể thực hiện các thử nghiệm A/B/C/D nếu bạn muốn.
Nhược điểm:
Không an toàn như Shadow Deployments: Đòi hỏi một đánh giá offline mạnh mẽ hơn để đảm bảo mô hình không gặp sự cố nếu bạn đưa lưu lượng thực tế qua nó.
Lựa chọn giữa rủi ro và mẫu đủ nhanh: Bạn phải chọn giữa giả định nhiều rủi ro hơn (chuyển nhiều lưu lượng đến mô hình B) và việc thu thập mẫu đủ nhanh để phân tích.

- Canary Release:Triển khai mô hình thách thức và mô hình hiện tại song song, nhưng bắt đầu với mô hình thách thức không nhận lưu lượng. Dần dần chuyển lưu lượng từ mô hình hiện tại sang mô hình thách thức (gọi là "canary"). Giám sát các thước đo hiệu suất của mô hình thách thức, nếu tất cả đều tốt, tiếp tục chuyển lưu lượng cho đến khi toàn bộ lưu lượng điều hướng đến mô hình thách thức.
Ưu điểm:
Dễ hiểu và triển khai: Chiến lược đơn giản nhất nếu bạn đã có cơ sở hạ tầng đặc điểm tính năng.
Quan sát phản ứng người dùng: Dự đoán từ mô hình thách thức được phục vụ, cho phép sử dụng với mô hình yêu cầu tương tác của người dùng.
Chi phí thấp: Giống như A/B Testing, chỉ có một dự đoán cho mỗi yêu cầu.
Linh tinh với A/B Testing: Có thể kết hợp với A/B testing để đo lường chính xác hiệu suất.
Nhược điểm:
Rủi ro không nghiêm túc về xác định sự chênh lệch hiệu suất: Có khả năng không nghiêm túc trong việc xác định sự chênh lệch hiệu suất giữa các mô hình.
Rủi ro tai nạn nếu triển khai không cẩn thận: Có thể xảy ra sự cố nếu triển khai không được giám sát cẩn thận, tuy nhiên, việc quay lại phiên bản trước đó (rollback) là khá dễ dàng.
