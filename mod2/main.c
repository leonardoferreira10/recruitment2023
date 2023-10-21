#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

int main() {
    const char *pipe_name = "/tmp/my_pipe4";  // Nome do pipe nomeado

    // Cria o pipe nomeado
    if (mkfifo(pipe_name, 0666) == -1) {
        perror("Erro ao criar o pipe nomeado");
        exit(1);
    }

    printf("Aguardando conexão do Python...\n");

    // Abre o pipe para leitura
    int pipe_fd = open(pipe_name, O_RDONLY);
    if (pipe_fd == -1) {
        perror("Erro ao abrir o pipe nomeado para leitura");
        exit(1);
    }

    printf("Conexão estabelecida com o Python. Aguardando dados...\n");

    char buffer[1024];  // Buffer para armazenar os dados lidos

    while (1) {
        ssize_t bytes_read = read(pipe_fd, buffer, sizeof(buffer));
        if (bytes_read <= 0) {
            printf("Conexão do Python encerrada.\n");
            break;
        }

        buffer[bytes_read] = '\0';
        printf("Resultado da inferência recebido: %s\n", buffer);
    }

    // Fecha o pipe e remove-o
    close(pipe_fd);
    unlink(pipe_name);

    return 0;
}
