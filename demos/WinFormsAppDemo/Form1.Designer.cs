namespace WinFormsAppDemo
{
    partial class Form1
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            richTextBox1 = new RichTextBox();
            textBox1 = new TextBox();
            button1 = new Button();
            button2 = new Button();
            label1 = new Label();
            textBox2 = new TextBox();
            label2 = new Label();
            button3 = new Button();
            button4 = new Button();
            label3 = new Label();
            textBox3 = new TextBox();
            button5 = new Button();
            textBox4 = new TextBox();
            label4 = new Label();
            button6 = new Button();
            label5 = new Label();
            pictureBox1 = new PictureBox();
            label6 = new Label();
            button7 = new Button();
            label7 = new Label();
            comboBox1 = new ComboBox();
            button8 = new Button();
            ((System.ComponentModel.ISupportInitialize)pictureBox1).BeginInit();
            SuspendLayout();
            // 
            // richTextBox1
            // 
            richTextBox1.Location = new Point(12, 490);
            richTextBox1.Name = "richTextBox1";
            richTextBox1.Size = new Size(792, 680);
            richTextBox1.TabIndex = 0;
            richTextBox1.Text = "";
            // 
            // textBox1
            // 
            textBox1.Font = new Font("Microsoft YaHei UI", 12F, FontStyle.Regular, GraphicsUnit.Point, 134);
            textBox1.Location = new Point(269, 100);
            textBox1.Name = "textBox1";
            textBox1.Size = new Size(496, 33);
            textBox1.TabIndex = 1;
            // 
            // button1
            // 
            button1.Font = new Font("Microsoft YaHei UI", 12F);
            button1.Location = new Point(269, 169);
            button1.Name = "button1";
            button1.Size = new Size(212, 44);
            button1.TabIndex = 2;
            button1.Text = "选择ONNX模型";
            button1.UseVisualStyleBackColor = true;
            button1.Click += button1_Click;
            // 
            // button2
            // 
            button2.Font = new Font("Microsoft YaHei UI", 12F);
            button2.Location = new Point(553, 169);
            button2.Name = "button2";
            button2.Size = new Size(212, 44);
            button2.TabIndex = 3;
            button2.Text = "转换ONN模型";
            button2.UseVisualStyleBackColor = true;
            button2.Click += button2_Click;
            // 
            // label1
            // 
            label1.AutoSize = true;
            label1.Font = new Font("Microsoft YaHei UI", 12F, FontStyle.Regular, GraphicsUnit.Point, 134);
            label1.Location = new Point(56, 103);
            label1.Name = "label1";
            label1.Size = new Size(158, 27);
            label1.TabIndex = 4;
            label1.Text = "ONNX模型路径:";
            // 
            // textBox2
            // 
            textBox2.Font = new Font("Microsoft YaHei UI", 12F, FontStyle.Regular, GraphicsUnit.Point, 134);
            textBox2.Location = new Point(1085, 100);
            textBox2.Name = "textBox2";
            textBox2.Size = new Size(496, 33);
            textBox2.TabIndex = 1;
            textBox2.Text = "E:\\Model\\yolo\\yolov8s.engine";
            // 
            // label2
            // 
            label2.AutoSize = true;
            label2.Font = new Font("Microsoft YaHei UI", 12F, FontStyle.Regular, GraphicsUnit.Point, 134);
            label2.Location = new Point(869, 103);
            label2.Name = "label2";
            label2.Size = new Size(161, 27);
            label2.TabIndex = 4;
            label2.Text = "Engine模型路径:";
            // 
            // button3
            // 
            button3.Font = new Font("Microsoft YaHei UI", 12F);
            button3.Location = new Point(1085, 169);
            button3.Name = "button3";
            button3.Size = new Size(212, 44);
            button3.TabIndex = 2;
            button3.Text = "选择Engine模型";
            button3.UseVisualStyleBackColor = true;
            button3.Click += button3_Click;
            // 
            // button4
            // 
            button4.Font = new Font("Microsoft YaHei UI", 12F);
            button4.Location = new Point(1369, 169);
            button4.Name = "button4";
            button4.Size = new Size(212, 44);
            button4.TabIndex = 2;
            button4.Text = "加载Engine模型";
            button4.UseVisualStyleBackColor = true;
            button4.Click += button4_Click;
            // 
            // label3
            // 
            label3.AutoSize = true;
            label3.Font = new Font("Microsoft YaHei UI", 12F, FontStyle.Regular, GraphicsUnit.Point, 134);
            label3.Location = new Point(876, 250);
            label3.Name = "label3";
            label3.Size = new Size(157, 27);
            label3.TabIndex = 4;
            label3.Text = "Image图片路径:";
            // 
            // textBox3
            // 
            textBox3.Font = new Font("Microsoft YaHei UI", 12F, FontStyle.Regular, GraphicsUnit.Point, 134);
            textBox3.Location = new Point(1085, 247);
            textBox3.Name = "textBox3";
            textBox3.Size = new Size(496, 33);
            textBox3.TabIndex = 1;
            textBox3.Text = "E:\\Data\\image\\bus.jpg";
            // 
            // button5
            // 
            button5.Font = new Font("Microsoft YaHei UI", 12F);
            button5.Location = new Point(1369, 304);
            button5.Name = "button5";
            button5.Size = new Size(212, 44);
            button5.TabIndex = 2;
            button5.Text = "模型推理";
            button5.UseVisualStyleBackColor = true;
            button5.Click += button5_Click;
            // 
            // textBox4
            // 
            textBox4.Font = new Font("Microsoft YaHei UI", 12F, FontStyle.Regular, GraphicsUnit.Point, 134);
            textBox4.Location = new Point(269, 244);
            textBox4.Name = "textBox4";
            textBox4.Size = new Size(496, 33);
            textBox4.TabIndex = 1;
            textBox4.Text = "1";
            // 
            // label4
            // 
            label4.AutoSize = true;
            label4.Font = new Font("Microsoft YaHei UI", 12F, FontStyle.Regular, GraphicsUnit.Point, 134);
            label4.Location = new Point(131, 251);
            label4.Name = "label4";
            label4.Size = new Size(97, 27);
            label4.TabIndex = 4;
            label4.Text = "测试轮次:";
            // 
            // button6
            // 
            button6.Font = new Font("Microsoft YaHei UI", 12F);
            button6.Location = new Point(269, 304);
            button6.Name = "button6";
            button6.Size = new Size(212, 44);
            button6.TabIndex = 2;
            button6.Text = "单线程速度测试";
            button6.UseVisualStyleBackColor = true;
            button6.Click += button6_Click;
            // 
            // label5
            // 
            label5.AutoSize = true;
            label5.Font = new Font("Microsoft YaHei UI", 18F, FontStyle.Bold, GraphicsUnit.Point, 134);
            label5.Location = new Point(484, 9);
            label5.Name = "label5";
            label5.Size = new Size(611, 40);
            label5.TabIndex = 4;
            label5.Text = "NVIDIA TensorRtSharp 推理工具测试平台";
            // 
            // pictureBox1
            // 
            pictureBox1.BackColor = SystemColors.ControlDark;
            pictureBox1.BackgroundImageLayout = ImageLayout.None;
            pictureBox1.Location = new Point(847, 490);
            pictureBox1.Name = "pictureBox1";
            pictureBox1.Size = new Size(878, 680);
            pictureBox1.TabIndex = 5;
            pictureBox1.TabStop = false;
            // 
            // label6
            // 
            label6.AutoSize = true;
            label6.Font = new Font("Microsoft YaHei UI", 14F, FontStyle.Bold, GraphicsUnit.Point, 134);
            label6.Location = new Point(299, 434);
            label6.Name = "label6";
            label6.Size = new Size(147, 31);
            label6.TabIndex = 4;
            label6.Text = "Logger日志";
            // 
            // button7
            // 
            button7.Font = new Font("Microsoft YaHei UI", 12F);
            button7.Location = new Point(1085, 304);
            button7.Name = "button7";
            button7.Size = new Size(212, 44);
            button7.TabIndex = 2;
            button7.Text = "选择Image图片";
            button7.UseVisualStyleBackColor = true;
            button7.Click += button7_Click;
            // 
            // label7
            // 
            label7.AutoSize = true;
            label7.Font = new Font("Microsoft YaHei UI", 14F, FontStyle.Bold, GraphicsUnit.Point, 134);
            label7.Location = new Point(1198, 434);
            label7.Name = "label7";
            label7.Size = new Size(110, 31);
            label7.TabIndex = 4;
            label7.Text = "推理结果";
            // 
            // comboBox1
            // 
            comboBox1.Font = new Font("Microsoft YaHei UI", 12F, FontStyle.Regular, GraphicsUnit.Point, 134);
            comboBox1.FormattingEnabled = true;
            comboBox1.Location = new Point(869, 169);
            comboBox1.Name = "comboBox1";
            comboBox1.Size = new Size(182, 35);
            comboBox1.TabIndex = 6;
            // 
            // button8
            // 
            button8.Font = new Font("Microsoft YaHei UI", 12F);
            button8.Location = new Point(553, 304);
            button8.Name = "button8";
            button8.Size = new Size(212, 44);
            button8.TabIndex = 2;
            button8.Text = "多线程速度测试";
            button8.UseVisualStyleBackColor = true;
            button8.Click += button8_Click;
            // 
            // Form1
            // 
            AutoScaleDimensions = new SizeF(9F, 20F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(1737, 1182);
            Controls.Add(comboBox1);
            Controls.Add(pictureBox1);
            Controls.Add(label7);
            Controls.Add(label6);
            Controls.Add(label4);
            Controls.Add(label3);
            Controls.Add(label2);
            Controls.Add(label5);
            Controls.Add(label1);
            Controls.Add(button2);
            Controls.Add(button4);
            Controls.Add(button8);
            Controls.Add(button6);
            Controls.Add(button5);
            Controls.Add(button7);
            Controls.Add(button3);
            Controls.Add(textBox4);
            Controls.Add(textBox3);
            Controls.Add(button1);
            Controls.Add(textBox2);
            Controls.Add(textBox1);
            Controls.Add(richTextBox1);
            Font = new Font("Microsoft YaHei UI", 9F, FontStyle.Regular, GraphicsUnit.Point, 134);
            Name = "Form1";
            Text = "Form1";
            Load += Form1_Load;
            ((System.ComponentModel.ISupportInitialize)pictureBox1).EndInit();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private RichTextBox richTextBox1;
        private TextBox textBox1;
        private Button button1;
        private Button button2;
        private Label label1;
        private TextBox textBox2;
        private Label label2;
        private Button button3;
        private Button button4;
        private Label label3;
        private TextBox textBox3;
        private Button button5;
        private TextBox textBox4;
        private Label label4;
        private Button button6;
        private Label label5;
        private PictureBox pictureBox1;
        private Label label6;
        private Button button7;
        private Label label7;
        private ComboBox comboBox1;
        private Button button8;
    }
}
